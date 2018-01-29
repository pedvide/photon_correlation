# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:11:05 2017

@author: villanueva
"""
import struct
import logging
import datetime
#import os

def iterate_header(data_file):
    logger = logging.getLogger('correlation.ptu_parser')

    # tag types
    tyEmpty8      = int('FFFF0008', 16)
    tyBool8       = int('00000008', 16)
    tyInt8        = int('10000008', 16)
    tyBitSet64    = int('11000008', 16)
    tyColor8      = int('12000008', 16)
    tyFloat8      = int('20000008', 16)
    tyTDateTime   = int('21000008', 16)
    tyFloat8Array = int('2001FFFF', 16)
    tyAnsiString  = int('4001FFFF', 16)
    tyWideString  = int('4002FFFF', 16)
    tyBinaryBlob  = int('FFFFFFFF', 16)

    # parse header
    magic_number = data_file.read(8).decode('ascii').strip('\x00').strip()
    logger.debug('File format: ' + magic_number)
    if 'PQTTTR' not in magic_number:
        logger.debug('ERROR, WRONG FILE FORMAT! ({})'.format(magic_number))
        raise ValueError('Wrong file format')
    version = data_file.read(8).decode('ascii')
    logger.debug('Version: ' + version)

    tag_id = ''
    tag_lists = {}
    while 'Header_End' not in tag_id:
         # identity, string
        tag_id = data_file.read(32).decode('ascii').strip('\x00').strip()
        # position in array, or -1 if not an array, int32
        tag_pos_array = struct.unpack('i', data_file.read(4))[0]
        # type of tag, uint32
        tag_type = struct.unpack('I', data_file.read(4))[0]

        if tag_type == tyEmpty8:  # empty value
            data_file.read(8)
            tag_value = ''
            logger.debug(tag_id + '(Empty)')
        elif tag_type == tyBool8:  # bool value
            tag_value = bool(struct.unpack('q', data_file.read(8))[0])
            logger.debug(tag_id + ' = tyBool8: {}'.format(tag_value))
        elif tag_type == tyInt8:  # integer
            tag_value = struct.unpack('q', data_file.read(8))[0]
            logger.debug(tag_id + ' = tyInt8: {}'.format(tag_value))
        elif tag_type == tyBitSet64:  # integer
            tag_value = struct.unpack('q', data_file.read(8))[0]
            logger.debug(tag_id + ' = tyBitSet64: {}'.format(tag_value))
        elif tag_type == tyColor8:  # RGB
            tag_value = struct.unpack('q', data_file.read(8))[0]
            logger.debug(tag_id + ' = tyColor8: {}'.format(tag_value))
        elif tag_type == tyFloat8:  # double
            tag_value = struct.unpack('d', data_file.read(8))[0]
            logger.debug(tag_id + ' = tyFloat8: {}'.format(tag_value))
        elif tag_type == tyTDateTime:  # datetime
            tag_days = struct.unpack('d', data_file.read(8))[0]
            tag_value = datetime.date(1899, 12, 30) + datetime.timedelta(days=tag_days)
            logger.debug(tag_id + ' = tyTDateTime: {}'.format(tag_value))
        elif tag_type == tyFloat8Array:  # array of float8
            tag_len = struct.unpack('q', data_file.read(8))[0]
            logger.debug(tag_id + ' = tyFloat8Array with {} entries'.format(tag_len))
        elif tag_type == tyAnsiString:  # ASCII string
            tag_len = struct.unpack('q', data_file.read(8))[0]
            tag_value = data_file.read(tag_len).decode('ascii').strip('\x00').strip()
            logger.debug(tag_id + ' = tyAnsiString: {}'.format(tag_value))
        elif tag_type == tyWideString:  # wide string
            tag_len = struct.unpack('q', data_file.read(8))[0]
            tag_value = data_file.read(tag_len).decode('ascii').strip('\x00').strip()
            logger.debug(tag_id + ' = tyWideString: {}'.format(tag_value))
        elif tag_type == tyBinaryBlob:  # binary data
            tag_len = struct.unpack('q', data_file.read(8))[0]
            tag_value = data_file.read(tag_len)
            logger.debug(tag_id + ' = tyBinaryBlob with {} bytes'.format(tag_len))
        else:
            logger.debug('Error, wrong tag type: {}, id: {}'.format(tag_type, tag_id))
            raise ValueError('Error, wrong tag type: {}, id: {}'.format(tag_type, tag_id))

        if tag_pos_array != -1:  # array element
            if tag_pos_array == 0:  # first array element
                tag_lists[tag_id] = []
            tag_lists[tag_id].append(tag_value)
            yield (tag_id, tag_lists[tag_id], data_file.tell())
        else:
            yield (tag_id, tag_value, data_file.tell())

    yield ('end_header_pos', data_file.tell(), data_file.tell())

def parse_header(filename):
    logger = logging.getLogger('correlation.ptu_parser')
    # (SubID = $00, RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $06 (TimeHarp260P)
    rtTimeHarp260PT2 = int('00010206', 16)

    with open(filename, 'rb') as data_file:
        header = {tag_id: tag_value for tag_id, tag_value, _ in iterate_header(data_file)}

    if header['TTResultFormat_TTTRRecType'] != rtTimeHarp260PT2:
        msg = 'File was not measured on a TimeHarp 260 Pico'
        logger.error(msg)
        raise ValueError(msg)
    return header

def get_from_header(file, key):
    for tag_id, tag_value, _ in iterate_header(file):
        if tag_id == key:
            return tag_value

# not used
#def fix_header(filename, new_items):
#    logger = logging.getLogger('correlation.ptu_parser')
#    with open(filename, 'r+b') as ptu_file:
#        for tag_id, tag_value, position in iterate_header(ptu_file):
#            if tag_id in new_items:
#                binary_data = struct.pack('q', new_items[tag_id])
#                write_position = position-len(binary_data)
#                logger.debug('Writing {} to {} at {}'.format(new_items[tag_id], tag_id, write_position))
#                ptu_file.seek(write_position, os.SEEK_SET)
#                ptu_file.write(binary_data)
