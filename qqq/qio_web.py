'''
module implementing classes to facilitate web scraping

Classes:
    HTMLCutter:
        this class can be though of as implementing "regular expressions++" on
        HTML documents. leveraging `re` and `lxml`, each instance carries a
        sequential set of instructions on how to extract useful information
        from an HTML document.

        the use case targeted is large volumes of similar html pages, likely
        to be encountered during web-scraping or crawling. an example:

            >>> wc = HTMLCutter([(HTMLCutter.XPATH, '//p[@class="main-text"]'),
            ...                  (HTMLCutter.STRIP, 'a'),
            ...                  (HTMLCutter.REGEX, '[A-Z][a-z]+')]
            >>> wc.cut(rqs.get('www.example.com').text)

        would extract a list of all Capitalized words contained inside
        <p class="main-text"> tags, including text wrapped in <a> tags.
'''
from queue import Queue
import re
from threading import Lock

from lxml.html import fromstring, etree
import requests as rqs

from .qio import Scraper


class HTMLCutter:
    '''
    cuts up an html document to extract desired information

    this is done by implementing a rudimentary "mini-language"
    consisting of opcodes and arguments to such opcodes to sequentially
    process a list of entities, starting with the singleton list
    consisting of the HTML document parsed with `lxml.html.fromstring'

    see `HTMLCutter.init` for documentation on opcodes

    '''
    # TODO: constants rather than magic strings
    XPATH = 'xpath'
    STRIP = 'strip'
    TEXT = 'text'
    REGEX = 'regex'
    SELECT = 'select'

    _XPATH_T = 'xpath_t'
    _XPATH_P = 'xpath_p'
    _REGEX_1G = 'regex_1g'
    _REGEX_NG = 'regex_ng'

    _ops = set([XPATH, STRIP, TEXT, REGEX, SELECT])

    T_ETREE = 'etree'
    T_UNICODE = 'unicode'
    T_TUP_UNICODE = 'tup_unicode'

    # TODO: constants rather than magic strings
    _sigs = {_XPATH_T: (T_ETREE, T_ETREE),
             _XPATH_P: (T_ETREE, T_UNICODE),
             STRIP: (T_ETREE, T_ETREE),
             TEXT: (T_ETREE, T_UNICODE),
             _REGEX_1G: (T_UNICODE, T_UNICODE),
             _REGEX_NG: (T_UNICODE, T_TUP_UNICODE),
             SELECT: (T_TUP_UNICODE, T_UNICODE)}

    def __init__(self, opseq):
        '''
        Arguments:
            opseq:
                list of sequentially-compatible opcodes. rudimentary
                validation is done to make sure each opcode can
                process the output of the previous, and that opcode
                arguments are valid.

                an opcode consists of a constant, given as all-caps
                class members, and an argument whose type depends on
                the opcode. the argument will be referred to as the
                `oparg`.

        Opcodes:
            HTMLCutter.XPATH:
                the state should be a list of `Etree` elements.
                the `oparg` takes a valid XPath. concatenates the
                return lists of `Etree.xpath` for each `Etree` element
                in the current state, unless the XPath selects for a 
                property, in which case state becomes a list of unicode
                strings.
                
                care should be taken when passing absolute ("//") paths
                because these are NOT implicitly made relative.
            HTMLCutter.STRIP:
                input state should be a list of `Etree` elements.
                applies `etree.strip_tags`. `oparg` should be a string
                corresponding to a tag.
            HTMLCutter.TEXT:
                the state should be a list of `Etree` elements. extracts
                the `.text` of each of these
            HTMLCutter.REGEX:
                the state should be unicode strings. applies the regexp
                given in the `oparg` to each. if it has two or more
                capturing groups, the state becomes a list of tuples of
                strings, else the state remains a list of unicode strings.
            HTMLCutter.SELECT:
                the state should be a list of tuples of strings. the
                `oparg` is an integer. the state becomes a list of unicode
                strings, each selected from index `oparg` from the tuple
                at its index in the input state list.
        '''
        self.opseq = opseq
        self._validate_opseq()

    def _validate_opseq(self):
        # check ops are valid
        cur_type = self.T_ETREE
        for opx, (op, arg) in enumerate(self.opseq):
            if op not in self._ops:
                raise ValueError('invalid op {} detected in opseq'.format(op))

            sig_op = op
            # polymorphism expansion
            if op == self.REGEX:
                arg = re.compile(arg)
                if arg.groups > 1:
                    sig_op = self._REGEX_NG
                else:
                    sig_op = self._REGEX_1G
                self.opseq[opx] = (op, re.compile(arg))

            elif op == self.XPATH:
                # TODO: fragile
                if arg.split('/')[-1].startswith('@'):
                    sig_op = self._XPATH_P
                else:
                    sig_op = self._XPATH_T

                test_x = fromstring('<html></html>')
                test_x.xpath(arg)

            elif op == self.SELECT:
                assert isinstance(arg, int)

            expect_type = self._sigs[sig_op][0]
            if cur_type != expect_type:
                raise TypeError('op {} acts on wrong type {}'
                                .format(op, expect_type))
            else:
                cur_type = self._sigs[sig_op][1]

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
            if op == self.XPATH:
                state = sum([s.xpath(arg) for s in state], [])
            elif op == self.STRIP:
                for s in state:
                    etree.strip_tags(s, arg)
            elif op == self.TEXT:
                state = [s.text for s in state]
            elif op == self.REGEX:
                state = sum([arg.findall(s)
                             for s in state
                             if s is not None], [])
            elif op == self.SELECT:
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
