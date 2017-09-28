'''
Module implementing classes to facilitate web scraping.
'''
import sys
from collections import Counter, defaultdict
from functools import partial
from heapq import heappop, heappush
from queue import PriorityQueue as PQ
from queue import Queue
from threading import Lock
from typing import (Any, Callable, Dict, Generic, Iterable, List, NewType,
                    Optional, Set, Tuple, TypeVar, Counter as CounterT)
import time

import regex
import requests.exceptions as rqe
from lxml.html import etree, fromstring, tostring
from requests import get as http_get, head as http_head
from requests import Response
from tldextract import extract as tldx

from .io import ConcurrentProcessor as CP, QueueReader
from .io import Controller, FIFOWorker, PipeWorker, PoolThrottle, QueueTee
from .log import get_logger
from .util import ensure_type

URL = NewType('URL', str)
UResp = Tuple[URL, Response]
T = TypeVar('T')
LOG = get_logger(__file__)
DEBUG = LOG.debug


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
        if init_state is not None:
            self.init_state = init_state

    def init_state(self, s):  # type: ignore
        try:
            return fromstring(s)
        except Exception as e:
            return []

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
        rx = regex.compile(regexp)
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
            if len(state) == 0:
                return state
            try:
                state = [op(s) for s in state]
                state = [s for s in state if s is not None]
                if isinstance(state[0], list):
                    state = sum(state, [])
            except Exception:
                LOG.error(f'invalid op {op} on state {state}, index {opx}')
                raise

        return state

    def __add__(self, hc: 'HTMLCutter') -> 'HTMLCutter':
        return HTMLCutter(init_state=lambda s: self.cut(s) + hc.cut(s))


class Requester:

    def __init__(
            self, *, headers=None, types=None,
            referer='https://www.google.com',
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/58.0.3029.110 Safari/537.36'),
            timeout=0.5,
            throttle_pool=10, throttle_window=0.1, **kwargs) -> None:

        self._done_init = False

        self.timeout = timeout
        self.types = {'text/html'} if types is None else types
        self.headers = {
            'referer': referer,
            'user-agent': user_agent,
        } if headers is None else headers
        self.kwargs = kwargs

        self.throttle_pool = throttle_pool
        self.throttle_window = throttle_window

        self._done_init = True
        self.__recompile_get()

    def __recompile_get(self) -> None:
        self._pool = PoolThrottle(
            pool=self.throttle_pool,
            window=self.throttle_window,
        )

        self._http_head = self._pool(partial(
            http_head, headers=self.headers, timeout=self.timeout,
            **self.kwargs,
        ))
        self._http_get = self._pool(partial(
            http_get, headers=self.headers, timeout=self.timeout,
            **self.kwargs,
        ))

        def get(url: URL) -> Optional[Response]:
            resp = self._http_head(url)

            # follow redirects
            ix = 3
            while resp.status_code == 301 and ix > 0:
                resp = self._http_head(resp.headers['location'])
                ix -= 1

            if 'content-type' in resp.headers and any([
                    ct in resp.headers['content-type']
                    for ct in self.types]):
                # mypy support for decorators isn't there yet
                return self._http_get(url)  # type: ignore
            return None

        self.get = get

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in {
                'types', 'headers', 'timeout',
                'throttle_pool', 'throttle_window'}:
            if self._done_init:
                self.__recompile_get()


class HTTPScraper(Controller, Generic[T]):
    '''
    Base class for scraping of HTTP based resources.

    Utilizes a callback-centric design. Built with the requests library
    and uses its Request object - no need to reinvent the wheel.
    '''

    def __init__(
        self, *,
        requester: Requester,
        payload_worker_factory: Callable[[], PipeWorker[UResp, T]],
        urls_q: Queue,
        n_scraper_workers: int=16,
        n_payload_workers: int=2,
        extra_headers=None,
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
            crawled_urls:
                set of URLs to be considered already visited. any URL contained
                in it will not be scraped.
        '''

        Controller.__init__(self)

        self.requester = requester
        self.urls_q = ensure_type(urls_q, Queue)

        self._scrape_processor: CP[URL, UResp] = CP(
            'scrape processor',
            input_q=self.urls_q,
            worker_factory=FIFOWorker.mk_factory(self._scrape),
            n_workers=n_scraper_workers,
            unpack=True, prio=True,
            output_limit=5, buffer_size=2,
        )

        self._payload_processor: CP[UResp, T] = CP(
            'payload processor',
            input_q=self._scrape_processor.output_q,
            worker_factory=payload_worker_factory,
            n_workers=n_payload_workers,
            output_limit=50,
        )

        self.output_q = self._payload_processor.output_q

        self.add_subordinate(self._payload_processor)
        self.add_subordinate(self._scrape_processor)

    def _scrape(self, url: URL) -> Optional[UResp]:
        try:
            resp = self.requester.get(url)
            if resp is None:
                return None
        except (rqe.RequestException, rqe.ConnectionError) as e:
            return None

        LOG.verbose(f'got {resp.status_code} from {url[:50]}')
        return (url, resp)


class URLPrioritizer(PipeWorker[UResp, URL]):
    '''
    Prioritizes URLs for scraping based according to given mode.

    Also responsible for excluding already seen URLs.
    '''

    def __init__(
            self, *,
            seen_urls: Set[URL], seen_urls_lock: Lock,
            domain_counter: CounterT[URL],
            url_func: Callable[[Response], Iterable[URL]],
            mode='breadth',
            **kwargs) -> None:

        self.mode = mode
        self.url_func = url_func

        self.seen_urls: Set[URL] = seen_urls
        self.seen_urls_lock = seen_urls_lock
        self.domain_counter = domain_counter

        self._url_heap: List[Tuple[int, URL]] = []

    @classmethod
    def mk_factory(
            cls, url_func,
            seen_urls=None,
            counter=None,
            lock=None,
            **kwargs) -> Callable[[], "URLPrioritizer"]:

        if seen_urls is None:
            seen_urls = set()
        if counter is None:
            counter = Counter()
        if lock is None:
            lock = Lock()

        def factory():
            return cls(
                seen_urls=seen_urls,
                seen_urls_lock=lock,
                domain_counter=counter,
                url_func=url_func,
                **kwargs)

        return factory

    def can_absorb(self):
        return True

    def absorb(self, uresp: UResp) -> None:
        url, resp = uresp
        t = time.time()
        new_urls = {*self.url_func(resp)}
        dt = time.time() - t
        if dt > 1:
            LOG.warning(f'took {dt:.3f} to cut url {url}')

        with self.seen_urls_lock:
            new_urls -= self.seen_urls
            self.seen_urls |= new_urls

        for new_url in new_urls:
            dom = tldx(new_url).domain
            self.domain_counter[dom] += 1
            prio = self.domain_counter[dom] + url.count('/')
            psh = (prio, new_url)
            heappush(self._url_heap, psh)
            self.seen_urls.add(new_url)

    def _emit(self) -> Optional[URL]:
        try:
            return heappop(self._url_heap)[1]
        except IndexError:
            raise PipeWorker.EmptyEmit


class CrawlSpec:
    '''
    Class managing cutters to run on a per-url basis.
    '''

    def __init__(self):
        self.cutter_bank: Dict[Any, Dict[str, HTMLCutter]] = defaultdict(dict)

    def add(self, urx: str, key: str, cutter: HTMLCutter) -> 'CrawlSpec':
        '''
        Adds cutter(s) to extract content on URLs matching urx and classify
        it under key.
        '''

        urx_re = regex.compile(urx)
        if urx_re in self.cutter_bank and key in self.cutter_bank[urx_re]:
            LOG.warning(f'Overwriting cutter for key {key} at urx {urx}!')
        if key == 'url':
            LOG.warning(
                'Overwriting the default key "url"! Source url '
                'will not be in output.'
            )

        self.cutter_bank[urx_re][key] = cutter
        return self

    def cut(self, uresp: UResp) -> Dict[str, Any]:
        '''
        Runs the stored filter bank against HTML given by `html_str`,
        deciding which filters to match based on `url`.
        '''
        url, resp = uresp
        html_str = resp.text

        output = {'url': url}
        for regexp, cutter_dict in self.cutter_bank.items():
            if regexp.findall(url):
                output.update({
                    key: cutter.cut(html_str)
                    for key, cutter in cutter_dict.items()
                })
        return output


class HTMLScraper(HTTPScraper[T], Generic[T]):
    '''
    Crawls HTML websites.
    '''

    def __init__(
        self, *,
        seed_urls: Iterable[URL],
        payload_worker_factory: Callable[[], PipeWorker[UResp, T]],
        url_factory: Callable[[], URLPrioritizer]=None,
        url_func=None,
        crawled_urls: Set[URL]=None,
        n_url_workers: int=3,
        url_q_size=10,
        **kwargs,
    ) -> None:
        '''
        Notation:
            URX: "URL regular expression", a regular expression matching
            certain crawlable URLs.
        Arguments:
            seed_url:
                url from which crawling begins
            crawlspec:
                CrawlSpec instance to extract interesting payloads from data
                on a per-url basis.
            kwargs:
                keyword arguments to pass to the `qio.Scraper` instance used
                internally to fetch the web pages.
        '''

        self._reader = QueueReader(seed_urls, maxsize=url_q_size)

        HTTPScraper.__init__(
            self,
            urls_q=self._reader.output_q,
            payload_worker_factory=payload_worker_factory,
            **kwargs,
        )
        self.add_subordinate(self._reader)

        self._resp_tee: QueueTee[UResp] = QueueTee(
            input_q=self._scrape_processor.output_q,
            output_qs=2, maxsize=5,
        )
        self.add_subordinate(self._resp_tee)

        self._payload_processor.set_input_q(self._resp_tee[0])

        if url_factory is None:
            if url_func is None:
                raise ValueError(
                    'must provide a url extraction function '
                    'if using the default URLPrioritizer'
                )

            url_factory = URLPrioritizer.mk_factory(
                url_func=url_func,
            )

        self._url_processor: CP[UResp, URL] = CP(
            'url_processor',
            input_q=self._resp_tee[1],
            output_q=self.urls_q,
            worker_factory=url_factory,
            n_workers=n_url_workers,
            output_limit=5,
        )
        
        self.add_subordinate(self._url_processor)


class JSONScraper(HTTPScraper):
    '''
    Crawls a json API.
    '''

    def __init__(self, *, keys, **kwargs):
        pass
