package com.example.mlvstheworld


import android.graphics.Bitmap
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import com.example.mlvstheworld.ModelInputDims.modelScaleH
import com.example.mlvstheworld.ModelInputDims.modelScaleW
import com.example.mlvstheworld.databinding.ActivityMainBinding


enum class Models {
    BackgroundRemover, FreshnessClassifier
}

object ModelInputDims {
    const val modelScaleH = 176
    const val modelScaleW = 128

}


interface MLModelStats {
    val fileName: String
    val outputArray: Array<Array<Array<FloatArray>>>
    fun processOutput(
        context: AppCompatActivity,
        binding: ActivityMainBinding,
        outputArray: Array<Array<Array<FloatArray>>>
    )

    fun close(
        context: AppCompatActivity,
        binding: ActivityMainBinding
    )
}

object BackgroundRemover : MLModelStats {
    lateinit var thisBinding: ActivityMainBinding
    override val fileName: String = "BackgroundRemoverStatic2WxH_128x176"
    override val outputArray: Array<Array<Array<FloatArray>>> =
        arrayOf(Array(modelScaleH) {
            Array(modelScaleW) {
                floatArrayOf(1f)
            }
        })

    override fun processOutput(
        context: AppCompatActivity,
        binding: ActivityMainBinding,
        outputArray: Array<Array<Array<FloatArray>>>
    ) {
        thisBinding = binding
        val withAlpha = IntArray(modelScaleH * modelScaleW * 1) {
            Color.argb(
                (0.6f * 255).toInt(), 3,
                218, 197
            )
        }


        for ((i, ix) in outputArray[0].withIndex()) {
            for (j in ix.indices) {
                var alphaVal = 1f - outputArray[0][i][j][0]
                val cutoff = 0.6f
                alphaVal = if (alphaVal > cutoff) cutoff else alphaVal

                val pixelcolor = Color.argb(
                    (alphaVal * 255).toInt(), 3,
                    218, 197
                )

                withAlpha[i * modelScaleW + j + 0] = pixelcolor

            }
        }
//            Log.d("FirstFragment", "withAlpha ${withAlpha.contentToString()}")
        val alphaBitmap =
            Bitmap.createBitmap(modelScaleW, modelScaleH, Bitmap.Config.ARGB_8888, true)
        alphaBitmap.setPixels(withAlpha, 0, modelScaleW, 0, 0, modelScaleW, modelScaleH)


        context.runOnUiThread {
            binding.ivOutput.setImageBitmap(
                alphaBitmap
            )
        }


    }

    override fun close(context: AppCompatActivity, binding: ActivityMainBinding) {

        val cleanBitmap = Bitmap.createBitmap(
            modelScaleW,
            modelScaleH,
            Bitmap.Config.ARGB_8888,
            true
        )
        cleanBitmap.setPixels(
            IntArray(modelScaleW * modelScaleH * 1) { 255 },
            0,
            modelScaleW,
            0,
            0,
            modelScaleW,
            modelScaleH
        )

        context.runOnUiThread {
            binding.ivOutput.setImageBitmap(
                cleanBitmap
            )
        }

    }


}


object FreshnessClassifier : MLModelStats {
    override val fileName: String = "FreshnessFGclassifier2WxH_128x176"
    override val outputArray: Array<Array<Array<FloatArray>>> =
        arrayOf(Array(1) { Array(1) { FloatArray(12) } })

    override fun processOutput(
        context: AppCompatActivity,
        binding: ActivityMainBinding,
        outputArray: Array<Array<Array<FloatArray>>>
    ) {
        val labels = arrayListOf(
            "Fresh apple",
            "Fresh banana",
            "Fresh bitter gourd",
            "Fresh capsicum",
            "Fresh orange",
            "Fresh tomato",
            "Stale apple",
            "Stale banana",
            "Stale bitter gourd",
            "Stale capsicum",
            "Stale orange",
            "Stale tomato"
        )
        val smallOut = outputArray[0][0][0]
        val maxValIndex = smallOut.indexOfFirst { it == smallOut.maxOrNull() }

        context.runOnUiThread {
            binding.tvResultOutput.text = labels[maxValIndex]
        }
    }

    override fun close(context: AppCompatActivity, binding: ActivityMainBinding) {
        return
    }


}