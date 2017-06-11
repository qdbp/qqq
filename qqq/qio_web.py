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
            ...                  (HTMLCutter.TEXT, ''),
            ...                  (HTMLCutter.REGEX, '[A-Z][a-z]+')]
            >>> wc.cut(rqs.get('www.example.com').text)

        would extract a list of all Capitalized words contained inside
        <p class="main-text"> tags, including text wrapped in <a> tags.
'''
from collections import defaultdict
from enum import Enum, auto
from functools import partial
from queue import Queue
import re
import sys
from threading import Lock
from typing import Any, Callable, NewType, Tuple, Dict, Set, List

from lxml.html import fromstring, tostring, etree
import requests as rqs

from .qio import QueueProcessor, PoolThrottle


URL = NewType('URL', str)


class HTMLCutter:
    '''
    Extracts useful information from html documents.

    After instantiation, the operations to be performed on the html string
    are specified by chaining the methods of this class. These operations
    are then performed in order on the argument of `cut` each time `cut` is
    called.

    The opcode methods available are listed below.

    Opcodes:
        fun: an arbitrary callable which will be applied to every element of
            state. Should return the next value of state.
        meth: a method name, which will be invoked on each element of state.
        xp:
            the state should be a list of `Etree` elements.
            the `oparg` takes a valid XPath. concatenates the
            return lists of `Etree.xpath` for each `Etree` element
            in the current state, unless the XPath selects for a
            property, in which case state becomes a list of unicode
            strings.

            care should be taken when passing absolute ("//") paths
            because these are NOT implicitly made relative.
        strip:
            input state should be a list of `Etree` elements.
            applies `etree.strip_tags`, which removes the selected tag
            and flattens its content into its context. `oparg` should be a
            string corresponding to a tag.
        prune:
            input state should be a list of `Etree` elements.
            Applies `etree.strip_elements`, which deletes entirely every
            instance of tag and its subtree. `oparg` should be a string
            corresponding to the tag to prune.
        text:
            the state should be a list of `Etree` elements. extracts
            the `.text` of each of these. ignores the `oparg`.
        raw:
            takes `Etree` as argument and returns its string representation.
        re:
            the state should be unicode strings. applies the regexp
            given in the `oparg` to each. if it has two or more
            capturing groups, the state becomes a list of tuples of
            strings, else the state remains a list of unicode strings.
        sel:
            the state should be a list of tuples of strings. the
            `oparg` is an integer. the state becomes a list of unicode
            strings, each selected from index `oparg` from the tuple
            at its index in the input state list.
    '''

    class CutterOpError(Exception):
        pass  # noqa

    def __init__(self):
        self.opseq = []  # type: List[Callable[..., Any]]

    def fn(self, func: Callable):
        self.opseq.append(func)
        return self

    def meth(self, methname: str):
        self.opseq.append(lambda s: getattr(s, methname)())
        return self

    def prune(self, tag: str):
        self.opseq.append(lambda s: etree.strip_elements(s, tag) or s)
        return self

    def raw(self):
        self.opseq.append(lambda s: tostring(s, encoding='unicode'))
        return self

    def re(self, regexp: str):
        rx = re.compile(regexp)
        self.opseq.append(lambda s: rx.findall(s))
        return self

    def sel(self, ix: int):
        self.opseq.append(lambda s: s[ix])
        return self

    def strip(self, tag: str):
        self.opseq.append(lambda s: etree.strip_tags(s, tag) or s)
        return self

    def text(self):
        self.opseq.append(lambda s: s.text)
        return self

    def xp(self, xpath_spec: str):
        # test the spec
        fromstring('<html></html>').xpath(xpath_spec)
        self.opseq.append(lambda s: s.xpath(xpath_spec))
        return self

    def cut(self, html_str):
        state = [fromstring(html_str)]
        for opx, op in enumerate(self.opseq):
            if not state:
                return state
            try:
                state = [op(s) for s in state]
                state = [s for s in state if s is not None]
                if isinstance(state[0], list):
                    state = sum(state, [])
            except Exception:
                print(
                    f'invalid op {op} on state {state}, index {opx}',
                    file=sys.stderr,
                )
                raise
        return state


class CrawlSpec:
    '''
    Class managing cutters to run on a per-url basis.
    '''

    def __init__(self):
        self.filter_bank =\
            defaultdict(dict)  # type: Dict[Any, Dict[str, HTMLCutter]]

    def add(self, urx: str, cutter_dict: Dict[str, HTMLCutter]) -> None:
        urx_re = re.compile(urx)
        self.filter_bank[urx_re].update(cutter_dict)
        return self

    def cut(self, url, html_str):
        output = {}
        for regexp, cutter_dict in self.filter_bank.items():
            if regexp.findall(url):
                output.update({
                    key: cutter.cut(html_str)
                    for key, cutter in cutter_dict.items()
                })
        return output


class WebScraper:

    def __init__(
        self, *,
        seed_url: URL,
        url_cutter: HTMLCutter,
        crawl_spec: CrawlSpec,
        crawled: Set[URL]=None,
        requests_kwargs: Dict[str, Any]=None,
        scraper_workers: int=8,
        process_workers: int=2,
        throttle_pool=5,
        throttle_window=1,
    ) -> None:
        '''
        Notation:
            URX:
                "URL regular expression", a regular expression matching certain
                crawlable URLs.
        Arguments:
            seed_url:
                url from which crawling begins
            callbacks:
                a dictionary, keyed by URXs. each new URL will be matched to
                the most specific, as determined by pattern length, URX which
                matches it. the callback keyed by this URX will then be called
                on the HTML content fetched from the URL, provided it can be
                retrieved successfully.

                the keys of this dictionary determine which new URLs will
                be followed. keys mapping to `None` can be used to match URLs
                which should be followed, but whose content will be ignored
                except for finding further URLs.
            crawled:
                set of URLs to be considered already visited. any URL contained
                in it will not be scraped.
            kwargs:
                keyword arguments to pass to the `qio.Scraper` instance used
                internally to fetch the web pages.
        '''

        self.f_get = PoolThrottle(pool=throttle_pool, window=throttle_window)(
            partial(rqs.get, **(requests_kwargs or {}))
        )

        self.url_cutter = url_cutter
        self.crawl_spec = crawl_spec

        self.urls_q = Queue()  # type: Queue[URL]
        self.html_q = Queue()  # type: Queue[str]
        self.urls_q.put(seed_url)

        self._scrape_qp = QueueProcessor(
            input_q=self.urls_q,
            work_func=self._scrape,
            n_workers=scraper_workers,
        )

        self._dissect_qp = QueueProcessor(
            input_q=self._scrape_qp.output_q,
            work_func=self._dissect,
            n_workers=process_workers,
            output_limit=50,
        )

        self.output_q = self._dissect_qp.output_q

        if crawled is None:
            self.crawled = set()  # type: Set[URL]
        else:
            self.crawled = crawled

        self.crawled_lock = Lock()
        self.crawled.add(seed_url)

    def run(self):
        '''
        initiates the scraper.

        no requests are sent before this method is called
        '''
        self._scrape_qp.run()
        self._dissect_qp.run()

    def _scrape(self, url: URL):
        return (url, self.f_get(url).text)

    def _dissect(self, scrapeload: Tuple[URL, str]):
        url, html_str = scrapeload
        # print(f'called _scrape with html_str {html_str}')
        urls = self.url_cutter.cut(html_str)
        with self.crawled_lock:
            urls = {url for url in urls if url not in self.crawled}
            for url in urls:
                self.crawled.add(url)
                self.urls_q.put(url)

        out = self.crawl_spec.cut(url, html_str)
        if out:
            out['url'] = url
        return out
