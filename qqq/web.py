'''
Module implementing classes to facilitate web scraping.
'''
import re
import sys
from collections import defaultdict
from collections.abc import Collection  # type: ignore
from functools import partial
from queue import PriorityQueue, Queue
from threading import Lock
from typing import (Any, Callable, Dict, Generic, Iterable, List, NewType, Set,
                    Tuple, TypeVar, Union)

import requests as rqs
from lxml.html import etree, fromstring, tostring

from .io import PoolThrottle, QueueProcessor

URL = NewType('URL', str)
T = TypeVar('T')


class HTMLCutter:
    '''
    Extracts useful information from HTML documents.

    This class can be though of as implementing "regular expressions++" on
    HTML documents. leveraging `re` and `lxml`, each instance carries a
    sequential set of instructions on how to extract useful information
    from an HTML document.

    The use case targeted is large volumes of similar HTML pages, likely
    to be encountered during web-scraping or crawling

    Example:

        >>> hc = HTMLCutter()\\
        ...     .xp('//p[@class="main-text"]')\\
        ...     .strip("a")\\
        ...     .text()\\ 
        ...     .re(r"[A-Z][a-z]+")
        >>> hc.cut(rqs.get("www.example.com").text)

        The above code extracts a list of all Capitalized words contained
        inside <p class="main-text"> tags, including text wrapped in <a>
        tags.

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

    def __init__(self, init_state: Callable[[str], List[Any]]=None) -> None:
        self.opseq = []  # type: List[Callable[..., Any]]
        self.init_state = init_state or (lambda s: [fromstring(s)])

    def fn(self, func: Callable):
        self.opseq.append(func)
        return self

    def meth(self, methname: str):
        self.opseq.append(lambda s: getattr(s, methname)())
        return self

    def prune(self, tag: str):
        self.opseq.append(lambda s: etree.strip_elements(s, tag) or s)
        return self

    def raw(self, **kwargs):
        self.opseq.append(lambda s: tostring(s, encoding='unicode', **kwargs))
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

    def cut(self, html_str: str) -> List[Any]:
        state = self.init_state(html_str)
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

    def __add__(self, hc: 'HTMLCutter') -> 'HTMLCutter':
        return HTMLCutter(init_state=lambda s: self.cut(s) + hc.cut(s))


class CrawlSpec:
    '''
    Class managing cutters to run on a per-url basis.
    '''

    def __init__(self):
        self.filter_bank =\
            defaultdict(dict)  # type: Dict[Any, Dict[str, HTMLCutter]]

    def add(self, urx: str, cutter_dict: Dict[str, HTMLCutter]) -> 'CrawlSpec':
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


class WebScraper(Generic[T]):

    def __init__(
        self, *,
        seed: Union[URL, Iterable[URL]],
        payload_callback: Callable[[rqs.Request], T],
        url_callback: Callable[[rqs.Request], Iterable[URL]]=None,
        crawled: Collection[URL]=None,
        requests_kwargs: Dict[str, Any]=None,
        scraper_workers: int=8,
        process_workers: int=2,
        throttle_pool=1,
        throttle_window=0.2,
        referer='https://www.google.com',
        user_agent=(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.36'),
    ) -> None:

        '''
        Notation:
            URX:
                "URL regular expression", a regular expression matching certain
                crawlable URLs.

        Arguments:
            seed_generator:
                A finite iterator yielding a number of seed URLs from which to
                begin crawling. These URLs will be given lowest priority,
                so that a seed URL will only be scraped if no new URLs were
                added as a result of crawling preceding seeds.
            payload_callback:
                A callable invoked on every Request, returning an arbitrary
                output object, which will be collected in the `output_q`.
            url_callback:
                A callable invoked on every Request, returning an iterable
                of URLs. These will be added to the crawling queue and
                scraped in turn.
            crawled:
                set of URLs to be considered already visited. any URL contained
                in it will not be scraped.
        '''

        self.f_get = PoolThrottle(pool=throttle_pool, window=throttle_window)(
            partial(
                rqs.get, headers={
                    'referer': referer,
                    'user-agent': user_agent,
                },
                **(requests_kwargs or {}),
            )
        )

        self.url_callback = url_callback
        self.payload_callback = payload_callback

        self.urls_q = PriorityQueue()  # type: PriorityQueue
        self.html_q = Queue()  # type: Queue[str]

        self.crawled = crawled or set()
        self.crawled_lock = Lock()

        if not isinstance(seed, Iterable):
            seed = (seed,)
        for url in seed:
            self.urls_q.put((1, url))  # type: ignore
            self.crawled.add(url)  # type: ignore

        self._scrape_qp = QueueProcessor(
            input_q=self.urls_q,
            work_func=self._scrape,
            n_workers=scraper_workers,
            unpack=True, prio=True,
        )

        self._dissect_qp = QueueProcessor(
            input_q=self._scrape_qp.output_q,
            work_func=self._dissect,
            n_workers=process_workers,
            output_limit=50,
        )

        self.output_q = self._dissect_qp.output_q

    def run(self):
        '''
        initiates the scraper.

        no requests are sent before this method is called
        '''
        self._scrape_qp.run()
        self._dissect_qp.run()

    def _scrape(self, prio: int, url: URL):
        resp = self.f_get(url)
        return (url, resp.text)

    def _dissect(self, scrapeload: Tuple[URL, str]):
        url, html_str = scrapeload

        if self.url_callback is not None:
            urls = self.url_callback(url, html_str)
            with self.crawled_lock:
                for url in urls:
                    if url not in self.crawled:
                        self.crawled.add(url)
                        self.urls_q.put((0, url))

        return self.payload_callback(url, html_str)


class HTMLScraper(WebScraper):
    '''
    Crawls HTML websites.
    '''

    def __init__(
        self, *,
        seed_url: Union[URL, Iterator[URL]],
        crawl_spec: CrawlSpec,
        url_cutter: HTMLCutter=None,
    ) -> None:
        '''
        Notation:
            URX:
                "URL regular expression", a regular expression matching certain
                crawlable URLs.
        Arguments:
            seed_url:
                url from which crawling begins
            crawled:
                set of URLs to be considered already visited. any URL contained
                in it will not be scraped.
            kwargs:
                keyword arguments to pass to the `qio.Scraper` instance used
                internally to fetch the web pages.
        '''

        if not isinstance(seed_url, Iterator):
            seed_url = iter([seed_url])

        def url_callback(url, html_str):
            return url_cutter.cut(html_str)

        def payload_callback(url, html_str):
            return crawl_spec.cut(url, html_str)

        super().__init__(
            seed_generator=seed_url,
            url_callback=url_callback,
            payload_callback=payload_callback,
            **kwargs,
        )


class JSONScraper(WebScraper):
    '''
    Crawls a json API.
    '''

    def __init__(self, *, keys, **kwargs):
        pass
