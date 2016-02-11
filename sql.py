import os.path as osp
import re
import sqlite3 as sq3

PK_RE = re.compile(r'PRIMARY KEY\((.*?)\)')


class SQLiteDB:
    """ wrapper class around sq3ite databases, encapsulating
        operations common to any stateful database access """

    def __init__(self, db_fn, strict=False):
        self.db_fn = db_fn
        assert osp.isfile(db_fn)
        self.strict=strict

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
            # print(self.c.description)
            pk_raw = PK_RE.findall(out)[0]
        except IndexError:
            raise ValueError("no primary key in table {}".format(out))

        pk = tuple(t.strip() for t in pk_raw.split(','))
        return SQLiteTable(self.db_fn, self.conn, self.c, key, pk,
                           strict=self.strict)

    # TODO: deprecated
    def _insert_raw(self, tab, d, rep=True):
        repl = 'OR REPLACE ' if rep else ''
        ins = '(' + ','.join('?'*len(d)) + ')'
        exs = 'INSERT ' + repl + 'INTO {} VALUES'.format(tab) + ins
        self.c.execute(exs, d)


# TODO: NOT THREAD SAFE
# TODO: NOT THREAD SAFE
# TODO: NOT THREAD SAFE
class SQLiteTable:
    """
    Class managing a single table of a sqlite database.

    Exposes a dictionary-like interface, with tuples of the primary
    keys as keys, and the remaining columns as values
    """
    def __init__(self, db_fn, conn, c, tabn, pk,
                 strict=False):
        self.conn = conn
        self.c = c
        self.tabn = tabn
        self.pk = pk
        self.db_fn = db_fn
        self.strict = strict

        self.c.execute('SELECT * FROM {}'.format(self.tabn))
        self.colw = len(self.c.description)

        self._iter_list = iter([])

    def _row_from_kv(self, key, value):
        # TODO: arbitrarily-positioned pks
        return key + value

    def _check_row(self, row):
        assert len(row) == self.colw,\
            'row length (given {}) must correspond to number of columns in table ({})'\
            .format(len(row), self.colw)

    def _check_key(self, key):
        assert len(key) == len(self.pk),\
            'key length must correspond to primary key {}'.format(self.pk)

    def _getall(self):
        self.c.execute('SELECT * from {}'.format(self.tabn))

    def __getitem__(self, key):
        self._check_key(key)
        exe = ('SELECT * FROM {} WHERE '.format(self.tabn) +
               ' AND '.join('{}=:{}'.format(i, i)
                            for i in self.pk))
        self.c.execute(exe, dict(zip(self.pk, key)))
        return self.c.fetchall()

    def __setitem__(self, key, value):
        self.append(self._row_from_kv(key, value))

    def __delitem__(self, key):
        self._check_key(key)
        exe = ('DELETE FROM {} WHERE '.format(self.tabn) +
               ' AND '.join('{}=:{}'.format(i, i) for i in self.pk))
        self.c.execute(exe, dict(zip(self.pk, key)))

    def __add__(self, rows):
        assert len(rows) > 0
        for row in rows:
            try:
                self.append(row)
            except sq3.IntegrityError as e:
                if self.strict:
                    raise e
        return self

    def append(self, row):
        self._check_row(row)
        exe = ('INSERT OR REPLACE INTO {} VALUES ('.format(self.tabn)
               + ','.join(['?']*self.colw) + ')',
               row)
        self.c.execute(*exe)

    def __iter__(self):
        self._getall()
        self._iter_list = iter(self.c.fetchall())
        return self

    def __next__(self):
        return next(self._iter_list)

    def __len__(self):
        self._getall()
        return len(self.c.fetchall())

    def __str__(self):
        return "{}: TABLE {} ({} rows) [PRIMARY KEYS {}]"\
               .format(self.db_fn, self.tabn, len(self), self.pk)


if __name__ == "__main__":
    print('nonexistent table assert')
    try:
        with SQLiteDB('not_test.db') as tdb:
            print(tdb)
    except AssertionError:
        print('    pass')

    with SQLiteDB('test.db') as tdb:
        print('good db')

        print('bad table assert')
        try:
            t = tdb['test_table_kek']
        except ValueError:
            print('    pass')

        t = tdb['test_table_c']
        print('good table')
        
        print('bad append assert')
        try:
            t.append([(707, 707, 757, 'bad append')])
        except AssertionError:
            print('    pass')

        # check good row add
        t += [(99, 99, 'not bad words'),
              (101, 101, 'this might be pulled into a shared repo')]
        print('good add')

        # check bad row append
        print('wrong row length assert')
        try:
            t.append((55, 'failure'))
        except AssertionError:
            print('    pass')

        # check good row append
        t.append((777, 777, 'success'))
        print('good append')

        del t[(99, 99)]
        print('good delete')

        # check iteration
        for row in t:
            print(row)
        print('good iteration')
