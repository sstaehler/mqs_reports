#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Some scripts to read in the Mars BED extended QuakeML produced by the MQS
author: Fabian Euchner, Simon Stähler
'''
import numpy as np

from mqs_reports.event import Event


XMLNS_QUAKEML_BED = "http://quakeml.org/xmlns/bed/1.2"
XMLNS_QUAKEML_BED_MARS = "http://quakeml.org/xmlns/bed/1.2/mars"
XMLNS_QUAKEML_SST = "http://quakeml.org/xmlns/singlestation/1.0"
QML_EVENT_NAME_DESCRIPTION_TYPE = 'earthquake name'
QML_SINGLESTATION_PARAMETERS_ELEMENT_NAME = "singleStationParameters"
QML_MARSQUAKE_PARAMETERS_ELEMENT_NAME = "marsquakeParameters"

XMLNS_SINGLESTATION_ABBREV = "sst"
XMLNS_SINGLESTATION = "http://quakeml.org/xmlns/singlestation/1.0"


def lxml_prefix_with_namespace(elementname, namespace):
    """Prefix an XML element name with a namsepace in lxml syntax."""

    return "{{{}}}{}".format(namespace, elementname)


def lxml_text_or_none(element):
    """
    If an lxml element has a text node, return its value. Otherwise,
    return None.

    """

    txt = None
    try:
        txt = element.text

    except Exception:
        pass

    return txt


def qml_get_pick_time_for_phase(event_element, pref_ori_publicid, phase_name):
    # [contains(text(), '{}')]
    ph_el = event_element.findall(
        "./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori_publicid,
            lxml_prefix_with_namespace("arrival", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace("phase", XMLNS_QUAKEML_BED)))

    pick_time_str = ''
    pick_unc_str = ''

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

            ev_pick_unc = event_element.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("pick", XMLNS_QUAKEML_BED),
                    pickid.text,
                    lxml_prefix_with_namespace("time", XMLNS_QUAKEML_BED),
                    lxml_prefix_with_namespace("lowerUncertainty", XMLNS_QUAKEML_BED)))

            pick_time_str = str(ev_pick_time.text).strip()
            if ev_pick_unc is not None:
                pick_unc_str = str(ev_pick_unc.text).strip()
            else:
                pick_unc_str = ''
            break

    return pick_time_str, pick_unc_str


def qml_get_event_info_for_event_waveform_files(xml_root,
                                                location_quality,
                                                event_type,
                                                phase_list):
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

        if not lq.text[-1] in tuple(location_quality):
            continue

        # event type from mars extension
        mars_event_type = ev.find("./{}".format(
            lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED_MARS)))
        mars_event_type_str = str(mars_event_type.text).split(sep='#')[-1]

        if mars_event_type_str not in event_type:
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

        # Get single station origin (for PDF based distance and origin time)
        sso = qml_get_sso_info_for_event_element(xml_root=xml_root, ev=ev)
        if 'origin_time' in sso:
            sso_origin_time = sso['origin_time']
        else:
            sso_origin_time = None
        if 'distance' in sso:
            sso_distance = sso['distance']
            sso_distance_pdf = sso['distance_pdf']
        else:
            sso_distance = None
            sso_distance_pdf = None

        # Mars event type (from BED extension)
        mars_event_type_str = ''
        mars_event_type = ev.find("./{}".format(
            lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED_MARS)))

        if mars_event_type is not None:
            mars_event_type_str = str(mars_event_type.text).strip()

        picks = dict()
        picks_sigma = dict()
        for phase in phase_list:
            picks[phase], picks_sigma[phase] = qml_get_pick_time_for_phase(ev, pref_ori.text, phase)

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

        event_info.append(Event(
            name=ev_name,
            publicid=ev_publicid,
            origin_publicid=str(pref_ori.text).strip(),
            picks=picks,
            picks_sigma=picks_sigma,
            quality=str(lq.text).strip(),
            latitude=float(latitude),
            longitude=float(longitude),
            sso_distance=sso_distance,
            sso_distance_pdf=sso_distance_pdf,
            sso_origin_time=sso_origin_time,
            mars_event_type=mars_event_type_str,
            origin_time=origin_time))

    return event_info


def qml_get_sso_info_for_event_element(xml_root, ev):
    sso_info = {}

    # preferredOriginID
    preferred_ori_id = lxml_text_or_none(ev.find("./{}".format(
        lxml_prefix_with_namespace("preferredOriginID", XMLNS_QUAKEML_BED))))

    # find SingleStationOrigin that references preferredOrigin

    # this xpath expression is invalid. why?
    # bed_ori_ref = xml_root.xpath('./{}/{}/{}[text()="{}"]'.format(
    # lxml_prefix_with_namespace(
    # QML_SINGLESTATION_PARAMETERS_ELEMENT_NAME, XMLNS_SINGLESTATION),
    # lxml_prefix_with_namespace("singleStationOrigin", XMLNS_SINGLESTATION),
    # lxml_prefix_with_namespace("bedOriginReference", XMLNS_SINGLESTATION),
    # preferred_ori_id))

    bed_ori_ref = None

    for bed_ori_ref in xml_root.iterfind('./{}/{}/{}'.format(
            lxml_prefix_with_namespace(
                QML_SINGLESTATION_PARAMETERS_ELEMENT_NAME,
                XMLNS_SINGLESTATION),
            lxml_prefix_with_namespace("singleStationOrigin",
                                       XMLNS_SINGLESTATION),
            lxml_prefix_with_namespace("bedOriginReference",
                                       XMLNS_SINGLESTATION))):

        if bed_ori_ref.text.strip() == preferred_ori_id:
            break

    if bed_ori_ref is None:
        return sso_info

    sso = bed_ori_ref.getparent()

    # extract distance, origin time, depth, backazimuth
    pref_distance_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace(
            "preferredDistanceID", XMLNS_SINGLESTATION))))

    pref_ori_time_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace(
            "preferredOriginTimeID", XMLNS_SINGLESTATION))))

    pref_depth_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace("preferredDepthID",
                                   XMLNS_SINGLESTATION))))

    pref_azimuth_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace("preferredAzimuthID",
                                   XMLNS_SINGLESTATION))))

    if pref_distance_id is not None:
        distance = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    pref_distance_id,
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value",
                                               XMLNS_SINGLESTATION)
                    )))

        if distance is not None:
            sso_info['distance'] = float(distance)


        distance_pdf_variable = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    pref_distance_id,
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("pdf",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("variable",
                                               XMLNS_SINGLESTATION)
                )))
        distance_pdf_prob = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    pref_distance_id,
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("pdf",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("probability",
                                               XMLNS_SINGLESTATION)
                )))
        print(distance_pdf_variable)
        print(distance_pdf_prob)
        if distance_pdf_variable is not None:
            sso_info['distance_pdf'] = np.asarray((distance_pdf_variable, distance_pdf_prob),
                                                  dtype=float)

    if pref_ori_time_id is not None:
        origin_time = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace(
                        "originTime", XMLNS_SINGLESTATION),
                    pref_ori_time_id,
                    lxml_prefix_with_namespace(
                        "originTime", XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value", XMLNS_SINGLESTATION))))

        sso_info['origin_time'] = origin_time

    if pref_depth_id is not None:
        depth = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("depth", XMLNS_SINGLESTATION),
                    pref_depth_id,
                    lxml_prefix_with_namespace("depth", XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value", XMLNS_SINGLESTATION))))

        if depth is not None:
            sso_info['depth'] = float(depth)

    if pref_azimuth_id is not None:
        azimuth = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("azimuth", XMLNS_SINGLESTATION),
                    pref_azimuth_id,
                    lxml_prefix_with_namespace("azimuth", XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value", XMLNS_SINGLESTATION))))
        if azimuth is not None:
            sso_info['azimuth'] = float(azimuth)

    return sso_info


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
                                        'Pg', 'Sg', 'x1', 'x2', 'x3',
                                        'end'])
    print(test)
