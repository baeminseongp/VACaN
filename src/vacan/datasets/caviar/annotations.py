"""Streaming parser for CAVIAR CVML-style XML annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Annotation:
    frame: int
    object_id: str
    xc: float
    yc: float
    width: float
    height: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        return self.xc, self.yc + self.height / 2


def iter_annotations(path: Path):
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "frame":
            continue
        frame = int(element.attrib["number"])
        object_list = element.find("objectlist")
        if object_list is not None:
            for item in object_list.findall("object"):
                box = item.find("box")
                if box is None:
                    continue
                yield Annotation(
                    frame=frame,
                    object_id=item.attrib["id"],
                    xc=float(box.attrib["xc"]),
                    yc=float(box.attrib["yc"]),
                    width=float(box.attrib["w"]),
                    height=float(box.attrib["h"]),
                )
        element.clear()


def xml_frame_count(path: Path) -> int:
    count = 0
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "frame":
            count += 1
            element.clear()
    return count
