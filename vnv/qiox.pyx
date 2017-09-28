from libc.stdint cimport uint8_t, int32_t, uint32_t

import os.path as osp
import numpy as np
cimport numpy as np

cdef extern from "FLIF/flif.h":

    struct FLIF_IMAGE:
        pass

    struct FLIF_DECODER:
        pass

    FLIF_DECODER* flif_create_decoder()

    int32_t flif_decoder_decode_file(
        FLIF_DECODER* decoder,
        const char* filename,
    )
    FLIF_IMAGE* flif_decoder_get_image(
        FLIF_DECODER* decoder,
        size_t index,
    )

cdef extern from "FLIF/flif_common.h":

    # ctypedef struct FLIF_IMAGE FLIF_IMAGE

    uint32_t flif_image_get_width(FLIF_IMAGE* img)
    uint32_t flif_image_get_height(FLIF_IMAGE* img)
    void flif_image_read_row_RGBA8(
        FLIF_IMAGE* image,
        uint32_t row,
        void* buffer,
        size_t buffer_size_bytes
    )


cdef void read_whole_flif_image_rgba(FLIF_IMAGE *image,
                                     uint8_t *buf, 
                                     int width, int height):
    cdef int rx
    for rx in range(height):
        flif_image_read_row_RGBA8(
            image, rx, buf, width*4
        )
        buf += 4 * width


cpdef np.ndarray[uint8_t, ndim=3, mode="c"] flif_to_rgba_arr(str fn):
    
    with open(fn, 'rb') as f:
        pass

    cdef bytes bts = fn.encode('utf-8')
    cdef char* bytes_fn = bts

    cdef FLIF_DECODER *decoder = flif_create_decoder()
    flif_decoder_decode_file(decoder, bytes_fn)
    cdef FLIF_IMAGE *image = flif_decoder_get_image(decoder, 0)

    cdef int width = flif_image_get_width(image)
    cdef int height = flif_image_get_height(image)

    cdef np.ndarray[uint8_t, ndim=3, mode="c"] out_arr =\
        np.empty((height, width, 4), dtype=np.uint8) 

    read_whole_flif_image_rgba(image, <uint8_t *>out_arr.data, width, height)

    return np.array(out_arr)
