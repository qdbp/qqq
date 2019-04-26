"""
Module implementing classes to facilitate web scraping.
"""
from typing import Any, Callable
from typing import List, NewType, Tuple, TypeVar

from lxml.html import etree, fromstring, tostring
from requests import Response

import regex

from .log import get_logger

URL = NewType("URL", str)
UResp = Tuple[URL, Response]
T = TypeVar("T")
LOG = get_logger(__file__)
DEBUG = LOG.debug


class HTMLCutter:
    """
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
        xp: the state should be a list of `Etree` elements.
            the `oparg` takes a valid XPath. concatenates the
            return lists of `Etree.xpath` for each `Etree` element
            in the current state, unless the XPath selects for a
            property, in which case state becomes a list of unicode
            strings. Care should be taken when passing absolute ("//") paths
            because these are NOT implicitly made relative.
        strip: input state should be a list of `Etree` elements.
            applies `etree.strip_tags`, which removes the selected tag
            and flattens its content into its context. `oparg` should be a
            string corresponding to a tag.
        prune: input state should be a list of `Etree` elements.
            Applies `etree.strip_elements`, which deletes entirely every
            instance of tag and its subtree. `oparg` should be a string
            corresponding to the tag to prune.
        text: the state should be a list of `Etree` elements. extracts
            the `.text` of each of these. ignores the `oparg`.
        raw: takes `Etree` as argument and returns its string representation.
        re: the state should be unicode strings. applies the regexp
            given in the `oparg` to each. if it has two or more
            capturing groups, the state becomes a list of tuples of
            strings, else the state remains a list of unicode strings.
        sel: the state should be a list of tuples of strings. the
            `oparg` is an integer. the state becomes a list of unicode
            strings, each selected from index `oparg` from the tuple
            at its index in the input state list.
    """

    class CutterOpError(Exception):
        pass  # noqa

    def __init__(self, init_state: Callable[[str], List[Any]] = None) -> None:
        self.opseq = []  # type: List[Callable[..., Any]]
        if init_state is not None:
            self.init_state = init_state

    def init_state(self, s):  # type: ignore
        try:
            return fromstring(s)
        except Exception:
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
        self.opseq.append(lambda s: tostring(s, encoding="unicode", **kwargs))
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
        fromstring("<html></html>").xpath(xpath_spec)
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
                LOG.error(f"invalid op {op} on state {state}, index {opx}")
                raise

        return state

    def __add__(self, hc: "HTMLCutter") -> "HTMLCutter":
        return HTMLCutter(init_state=lambda s: self.cut(s) + hc.cut(s))
