from queue import Queue
import re
from threading import Lock

from lxml.html import fromstring, etree
import requests as rqs

from .qio import Scraper


class HTMLCutter:
    '''
    cuts up a top-level xml document to extract desired information
    '''
    # TODO: constants rather than magic strings
    ops = {'xpath': ['xp'],
           'strip': ['st'],
           'text': ['tx'],
           'regex': ['regexp', 're'],
           'select': ['sel']}

    # TODO: constants rather than magic strings
    sigs = {'xpath_tree': ('etree', 'etree'),
            'xpath_prop': ('etree', 'unicode'),
            'strip': ('etree', 'etree'),
            'text': ('etree', 'unicode'),
            'regex_1g': ('unicode', 'unicode'),
            'regex_ng': ('unicode', 'tup_unicode'),
            'select': ('tup_unicode', 'unicode')}

    def __init__(self, opseq):
        self.opseq = opseq
        self._validate_opseq()

    def _validate_opseq(self):
        new_opseq = []

        # canonicalize ops
        for op, arg in self.opseq:
            c_op = None
            for k, v in self.ops.items():
                if op == k or op in v:
                    c_op = k
                    break
            if c_op is None:
                raise ValueError('invalid op {} detected in opseq'.format(op))
            new_opseq.append((c_op, arg))
            self.opseq = new_opseq

        # validate types
        cur_type = 'etree'
        for opx, (c_op, arg) in enumerate(self.opseq):
            # polymorphism expansion
            if c_op == 'regex':
                arg = re.compile(arg)
                if arg.groups > 1:
                    sig_op = 'regex_ng'
                else:
                    sig_op = 'regex_1g'
                self.opseq[opx] = (c_op, re.compile(arg))
            elif c_op == 'xpath':
                # TODO: fragile
                if arg.split('/')[-1].startswith('@'):
                    sig_op = 'xpath_prop'
                else:
                    sig_op = 'xpath_tree'
            else:
                sig_op = c_op

            # basic argument validation
            if c_op == 'select':
                assert isinstance(arg, int)
            if c_op == 'xpath':
                x = fromstring('<html></html>')
                # should crash with malformed xpath
                x.xpath(arg)

            expect_type = self.sigs[sig_op][0]
            if cur_type != expect_type:
                raise TypeError('op {} acts on wrong type {}'
                                .format(c_op, expect_type))

            cur_type = self.sigs[sig_op][1]

    # TODO: preemptive opseq validation
    def cut(self, s):
        '''
        applies the opseq this instance was instantiated with to doc

        Arguments:
            s:
                html document, as string

        Returns:
            state:
                list of strings or `etree` nodes, determined by the
                `opseq`
        '''
        x = fromstring(s)
        state = [x]
        for op, arg in self.opseq:
            if op == 'xpath':
                state = sum([s.xpath(arg) for s in state], [])
            elif op == 'strip':
                for s in state:
                    etree.strip_tags(s, arg)
            elif op == 'text':
                state = [s.text for s in state]
            elif op == 'regex':
                state = sum([arg.findall(s)
                             for s in state
                             if s is not None], [])
            elif op == 'select':
                state = [s[arg] for s in state]
            else:
                raise ValueError('invalid op {}'.format(op))

        return state


class WebScraper:
    def __init__(self, seed_url, mining_rig, scraped_file='scraped.p', **kwargs):
        '''
        Arguments:
            seed_url:
                url from which crawling begins
            mining_rig:
                dictionary, {urx: None|{key: HTMLCutters}}.

                URLs matched by the keys of this dictionary will be crawled.
                furthermore, those mapping to non-`None`/empty values will
                be passed to each HTMLCutter in the
                inner dictionary, generating an output dictionary with the
                same keys as the cutter dictionary. the url scraped and the
                result of `htmlcutter.cut` called on the received contents of
                the matching url will be put in the output queue

                in the case where a url found on the page matches more than
                one pattern in `mining_rig.keys`, the most specific pattern,
                determined by the length of the `urx` regular expression, will
                be used. where two patterns have the same length, the one used
                is arbitrary.
            **kwargs:
                keyword arguments to pass to the `qio.Scraper` instance used
                internall to fetch the web pages.
        '''

        self.rig = {re.compile(k): v for k, v in mining_rig.items()}

        self.urls_q = Queue()
        self.urls_q.put((seed_url, self.rig[self._get_cutter(seed_url)]))

        self.output_q = Queue()

        self._scr = Scraper(self.urls_q, self._crawl, output_q=self.output_q,
                            **kwargs)

        self.crawled = set()
        self.crawled.add(seed_url)
        self._crl_lock = Lock()

    def run(self):
        self._scr.run()

    def _get_cutter(self, url):
        for urx in sorted(self.rig.keys(), key=lambda x: -len(x.pattern)):
            if urx.match(url):
                return urx

    def _add_urls(self, urls, cutter_dict):
        with self._crl_lock:
            for url in urls:
                if url not in self.crawled:
                    self.crawled.add(url)
                    self.urls_q.put((url, cutter_dict))

    def _crawl(self, url_cutter_dict):
        url, cutter_dict = url_cutter_dict

        html = rqs.get(url).text
        for urx in sorted(self.rig.keys(), key=lambda x: -len(x.pattern)):
            self._add_urls(urx.findall(html), self.rig[urx])

        if cutter_dict:
            out = {k: v.cut(html) for k, v in cutter_dict.items()}
            return (url, out)
