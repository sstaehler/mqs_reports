#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Some scripts to read in the Mars BED extended QuakeML produced by the MQS
author: Fabian Euchner, Simon Stähler
'''
XMLNS_QUAKEML_BED = "http://quakeml.org/xmlns/bed/1.2"
XMLNS_QUAKEML_BED_MARS = "http://quakeml.org/xmlns/bed/1.2/mars"
QML_EVENT_NAME_DESCRIPTION_TYPE = 'earthquake name'

from mqs_reports.event import Event
from obspy import UTCDateTime as utct


def lxml_prefix_with_namespace(elementname, namespace):
    """Prefix an XML element name with a namsepace in lxml syntax."""

    return "{{{}}}{}".format(namespace, elementname)


def qml_get_pick_time_for_phase(event_element, pref_ori_publicid, phase_name):
    # [contains(text(), '{}')]
    ph_el = event_element.findall(
        "./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori_publicid,
            lxml_prefix_with_namespace("arrival", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace("phase", XMLNS_QUAKEML_BED)))

    pick_time_str = ''

    for ph in ph_el:
        if str(ph.text).strip() == phase_name:
            pickid = ph.find(
                "../{}".format(
                    lxml_prefix_with_namespace("pickID", XMLNS_QUAKEML_BED)))

            ev_pick_time = event_element.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("pick", XMLNS_QUAKEML_BED),
                    pickid.text,
                    lxml_prefix_with_namespace("time", XMLNS_QUAKEML_BED),
                    lxml_prefix_with_namespace("value", XMLNS_QUAKEML_BED)))

            pick_time_str = str(ev_pick_time.text).strip()
            break

    return pick_time_str


def qml_get_event_info_for_event_waveform_files(xml_root, location_quality,
                                                event_type,
                                                phase_list=['start', 'end']):
    event_info = []

    for ev in xml_root.iter(
            "{}".format(
                lxml_prefix_with_namespace("event", XMLNS_QUAKEML_BED))):

        # publicID
        ev_publicid = ev.get("publicID")

        # preferred origin
        pref_ori = ev.find(
            "./{}".format(lxml_prefix_with_namespace(
                "preferredOriginID", XMLNS_QUAKEML_BED)))

        # location quality from mars extension
        lq = ev.find(
            "./{}[@publicID='{}']/{}".format(
                lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
                pref_ori.text,
                lxml_prefix_with_namespace(
                    "locationQuality", XMLNS_QUAKEML_BED_MARS)))

        # if not lq.text.endswith(tuple(location_quality)):
        #     continue
        if not lq.text[-1] in tuple(location_quality):
            continue

        # event type from mars extension
        mars_event_type = ev.find("./{}".format(
            lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED_MARS)))
        mars_event_type_str = str(mars_event_type.text).split(sep='#')[-1]

        if not mars_event_type_str in event_type:
            continue

        # event name
        desc_texts = ev.findall("./{}/{}".format(
            lxml_prefix_with_namespace("description", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace("text", XMLNS_QUAKEML_BED)))

        ev_name = ''
        for desc_text in desc_texts:

            desc_type = desc_text.find("../{}".format(
                lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED)))

            if str(desc_type.text).strip() == QML_EVENT_NAME_DESCRIPTION_TYPE:
                ev_name = str(desc_text.text).strip()
                break

        if not ev_name:
            continue

        # Mars event type (from BED extension)
        mars_event_type_str = ''
        mars_event_type = ev.find("./{}".format(
            lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED_MARS)))

        if mars_event_type is not None:
            mars_event_type_str = str(mars_event_type.text).strip()

        picks = dict()
        for phase in phase_list:
            picks[phase] = qml_get_pick_time_for_phase(ev, pref_ori.text,
                                                       phase)

        latitude = ev.find("./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori.text,
            lxml_prefix_with_namespace(
                "latitude", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace(
                "value", XMLNS_QUAKEML_BED))).text
        longitude = ev.find("./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori.text,
            lxml_prefix_with_namespace(
                "longitude", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace(
                "value", XMLNS_QUAKEML_BED))).text
        origin_time = ev.find("./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori.text,
            lxml_prefix_with_namespace(
                "time", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace(
                "value", XMLNS_QUAKEML_BED))).text


        if not ev_name == 'S0085a':
            event_info.append(Event(
                name=ev_name,
                publicid=ev_publicid,
                origin_publicid=str(pref_ori.text).strip(),
                picks=picks,
                quality=str(lq.text).strip(),
                latitude=float(latitude),
                longitude=float(longitude),
                mars_event_type=mars_event_type_str,
                origin_time=utct(origin_time)))

    return event_info


def read_QuakeML_BED(fnam, event_type, phase_list,
                     quality=('A', 'B', 'C', 'D')):
    from lxml import etree
    with open(fnam) as fh:
        tree = etree.parse(fh)
        xml_root = tree.getroot()
        events = qml_get_event_info_for_event_waveform_files(
                xml_root, location_quality=quality,
                event_type=event_type,
                phase_list=phase_list)

    return events


if __name__ == '__main__':
    test = read_QuakeML_BED('./mqs_reports/data/catalog_20191002.xml',
                            event_type='BROADBAND',
                            phase_list=['P', 'S', 'noise_start', 'start',
                                        'end'])
    print(test)
