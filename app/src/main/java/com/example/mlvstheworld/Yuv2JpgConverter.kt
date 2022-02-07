package com.example.mlvstheworld

import android.graphics.ImageFormat.NV21
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import java.io.ByteArrayOutputStream

object Converter {

    fun jpegByteArrayFrom(yuv420_888: Image): ByteArray =
        yuv420_888.nv21ByteArray
            .let { YuvImage(it, NV21, yuv420_888.width, yuv420_888.height, null) }
            .getJpegDataWithQuality(100)

    private val Image.nv21ByteArray
    // the size was orignially width * height * 3 / 2
        get() = ByteArray(planes[0].buffer.capacity() + planes[1].buffer.capacity()
                + planes[2].buffer.capacity()).also {
            Log.d("yuvConverter", "0 ${planes[0].buffer.capacity()}")
            Log.d("yuvConverter", "1 ${planes[1].buffer.capacity()}")
            Log.d("yuvConverter", "2 ${planes[2].buffer.capacity()}")
            Log.d("yuvConverter", "2 ${it.size}")
            Log.d("yuvConverter", "width ${width}")
            Log.d("yuvConverter", "height ${height}")

            val vPlane = planes[2]
            val y = planes[0].buffer.apply { rewind() }
            val u = planes[1].buffer.apply { rewind() }
            val v = vPlane.buffer.apply { rewind() }


            y.get(it, 0, y.capacity()) // copy Y components
            if (vPlane.pixelStride == 2) {
                // Both of U and V are interleaved data, so copying V makes VU series but last U
                v.get(it, y.capacity(), v.capacity())
                it[it.size - 1] = u.get(u.capacity() - 1) // put last U
            } else { // vPlane.pixelStride == 1
                var offset = it.size - 1
                var i = v.capacity()
                while (i-- != 0) { // make VU interleaved data into ByteArray
                    it[offset - 0] = u[i]
                    it[offset - 1] = v[i]
                    offset -= 2
                }
            }
        }

    private fun YuvImage.getJpegDataWithQuality(quality: Int) =
        ByteArrayOutputStream().also {
            compressToJpeg(Rect(0, 0, width, height), quality, it)
        }.toByteArray()

}