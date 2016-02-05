import os.path as osp
import re
import sqlite3 as sq3

PK_RE = re.compile(r'PRIMARY KEY\((.*?)\)')


class SQLiteDB:
    """ wrapper class around sq3ite databases, encapsulating
        operations common to any stateful database access """

    def __init__(self, db_fn):
        self.db_fn = db_fn
        assert osp.isfile(db_fn)

    def __enter__(self):
        self.conn = sq3.connect(self.db_fn)
        self.c = self.conn.cursor()
        return self

    def __exit__(self, _a, _b, _c):
        self.conn.commit()
        self.conn.close()

    def __getitem__(self, key):
        t = self.c.execute("SELECT * FROM sqlite_master WHERE type='table' "
                           "AND name=:key", {'key': key})
        try:
            out = t.fetchall()[0][-1]
        except IndexError:
            raise ValueError('failed to find table {} in {}'
                             .format(key, self.db_fn))

        try:
            print(self.c.description)
            pk_raw = PK_RE.findall(out)[0]
        except IndexError:
            raise ValueError("no primary key in table {}".format(out))
        
        pk = tuple(pk_raw.split(','))
        return SQLiteTable(self.conn, self.c, key, pk)

    def _insert_raw(self, tab, d, rep=True):
        repl = 'OR REPLACE ' if rep else ''
        ins = '(' + ','.join('?'*len(d)) + ')'
        exs = 'INSERT ' + repl + 'INTO {} VALUES'.format(tab) + ins
        self.c.execute(exs, d)


class SQLiteTable:
    def __init__(self, conn, c, tabn, pk, l):
        self.conn = conn
        self.c = c
        self.tabn = tabn
        self.pk = pk

    def _check_key(self, key):
        if not len(key) == len(self.pk):
            raise ValueError('key length must correspond to primary key {}'
                             .format(self.pk))

    def __getitem__(self, key):
        self._check_key(key)
        exe = ('SELECT * FROM {} WHERE '.format(self.tabn) +
               ' AND '.join('{}=:{}'.format(i, i)
                            for i in self.pk))
        self.c.execute(exe, dict(zip(self.pk, key)))
        return self.c.fetchall()

    def __setitem__(self, key, value):
        pass 
        # self._check_key(key)
        # exe = ('INSERT OR REPLACE INTO {} VALUES '.format(self.tabn) +
        #        ' AND '.join('{}=:{}'.format(i, i)
        #                     for i in self.pk))
        # pass


if __name__ == "__main__":
    try:
        with SQLiteDB('not_test.db') as tdb:
            print(tdb)
    except AssertionError:
        pass
    with SQLiteDB('test.db') as tdb:
        t = tdb['test_table_a']
        out = t[(3,)]
        print(out)
