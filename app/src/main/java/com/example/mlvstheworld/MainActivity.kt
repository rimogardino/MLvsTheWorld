package com.example.mlvstheworld

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.media.Image
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Menu
import android.view.MenuItem
import android.view.Surface.*
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.mlvstheworld.databinding.ActivityMainBinding
import com.google.common.util.concurrent.ListenableFuture
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val PERMISSIONS_REQUIRED = arrayOf(Manifest.permission.CAMERA)
    private lateinit var drawerToggle: ActionBarDrawerToggle
    private val strategyBackgroundRemover = "backgroundRemover"
    private val strategyFreshnessClassifier = "freshnessclassifier"
    private var modelStrategy: String = strategyBackgroundRemover
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService
    private var frameCount = 0


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        drawerToggle = ActionBarDrawerToggle(
            this, binding.drawerLayout,
            R.string.openDrawer,
            R.string.closeDrawer
        )
        binding.drawerLayout.addDrawerListener(drawerToggle)
        drawerToggle.syncState()

        // Navigation drawer side menu set up
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        binding.navViewMlList.setNavigationItemSelectedListener {
            when (it.itemId) {
                R.id.model_background_remover -> modelStrategy = strategyBackgroundRemover
                R.id.model_fresh_class -> {
                    modelStrategy = strategyFreshnessClassifier
                }

            }
            binding.drawerLayout.closeDrawers()
            true

        }


        // Request camera permissions
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this, PERMISSIONS_REQUIRED, 10
            )
        }


        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider)
        }, ContextCompat.getMainExecutor(this))


    }


    @SuppressLint("UnsafeOptInUsageError")
    private fun bindPreview(cameraProvider: ProcessCameraProvider) {
        val preview: Preview = Preview.Builder()
            .build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(binding.previewView.surfaceProvider)


        val modelScaleH = 176
        val modelScaleW = 128


        val imageAnalysis = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
//            .setTargetRotation(ROTATION_90)
            .setTargetResolution(Size(1280, 960))
            .setTargetRotation(ROTATION_90)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        cameraExecutor = Executors.newSingleThreadExecutor()

        imageAnalysis.setAnalyzer(cameraExecutor) { image ->
//            val rotationDegrees = image.imageInfo.rotationDegrees
            frameCount += 1
            Log.d("FirstFragment", "preview info ${preview.resolutionInfo?.resolution}")
            val planes: Array<out Image.Plane> = image.image!!.planes
            val buffer: ByteBuffer = planes[0].buffer

            val pixelStride: Int = planes[0].pixelStride
            val rowStride: Int = planes[0].rowStride

            val rowPadding = rowStride - pixelStride * image.width

            val bitmapW = image.width + rowPadding / pixelStride

            val bitmap = Bitmap.createBitmap(
                bitmapW,
                image.height, Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)
            Log.d(
                "FirstFragment",
                "binding.previewView.width  image.height ${binding.previewView.width / 128} ${binding.previewView.height}"
            )

            // Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.


//            val size: Int = if (image.height > image.width) image.width else image.height
//            val sizeH = modelScaleH * (image.height / modelScaleH)
//            val sizeW = modelScaleW * (bitmapW / modelScaleW)
//            Log.d("FirstFragment", "sizeW  sizeH ${sizeH} ${sizeH}")

            val imageProcessor = ImageProcessor.Builder()
                .add(Rot90Op(135))
//                .add(ResizeWithCropOrPadOp(sizeH, sizeW))
                .add(ResizeOp(modelScaleH, modelScaleW, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .build()

            // Create a TensorImage object. This creates the tensor of the corresponding
            // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
            val tensorImage = TensorImage(DataType.UINT8)

            // Analysis code for every frame
            // Preprocess the image
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            Log.d(
                "FirstFragment",
                "processedImage.width  processedImage.height ${processedImage.width} ${processedImage.height}"
            )
            // Create a container for the result and specify that this is a quantized model.
            // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
            //            val probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 7), DataType.UINT8)
            val tfliteOptions = Interpreter.Options()


            // Initialize interpreter with NNAPI delegate for Android Pie or above
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                Log.d("GPUFirstFragment", "Running on GPU")
                tfliteOptions.addDelegate(NnApiDelegate())
            } else {
                tfliteOptions.setNumThreads(4)
            }


            // Initialise the model
            var tflite: Interpreter? = null
            try {
                val modelString = when (modelStrategy) {
                    strategyBackgroundRemover -> {
                        this.runOnUiThread {
                            binding.tvResultOutput.text = getString(R.string.background_remover)
                        }

                        "BackgroundRemoverStatic2WxH_128x176"
                    }

                    strategyFreshnessClassifier -> {
                        this.runOnUiThread {
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
                            binding.ivOutput.setImageBitmap(
                                cleanBitmap
                            )
                        }

                        "FreshnessFGclassifier2WxH_128x176"
                    }
                    else -> "BackgroundRemover2StaticWxH_128x176"
                }


                val mappedByteBuffer = FileUtil.loadMappedFile(
                    this,
                    "$modelString.tflite"//BackgroundRemoverStaticOvertrained128x96 FreshnessFGclassifier128x128
                )


                tflite = Interpreter(mappedByteBuffer, tfliteOptions)
            } catch (e: IOException) {
                Log.e("tfliteSupport", "Error reading model", e)
            } catch (e: IllegalStateException) {
                Log.e("tfliteSupport", "rotation err", e)
            }

// Running inference

            val inputBuffer =
                arrayOf(Array(modelScaleH) { Array(modelScaleW) { floatArrayOf(0f, 0f, 0f) } })

            val outputArray = when (modelStrategy) {
                strategyBackgroundRemover -> {
                    Log.d("FirstFragment", "outputArray is $strategyBackgroundRemover")
                    arrayOf(Array(modelScaleH) { Array(modelScaleW) { floatArrayOf(1f) } })
                }

                strategyFreshnessClassifier -> {
                    Log.d("FirstFragment", "outputArray is $strategyFreshnessClassifier")
                    arrayOf(Array(1) { Array(1) { FloatArray(12) } })
                }
                else -> arrayOf(Array(modelScaleH) { Array(modelScaleW) { floatArrayOf(1f) } })
            }
            Log.d(
                "FirstFragment",
                "outputArray is ${outputArray[0].size} ${outputArray[0][0].size}"
            )


            val processedBitmap = processedImage.bitmap

            Log.d(
                "FirstFragment",
                "processedBitmap.height , width ${processedBitmap.height}, ${processedBitmap.width}"
            )

//            val testInput = IntArray(128 * 128 * 1) { 255 }

            for ((i, ix) in inputBuffer[0].withIndex()) {
                for (j in ix.indices) {
                    val pixelval = processedBitmap.getPixel(j, i)
//                    testInput[i * 128 + j + 0] = pixelval
                    inputBuffer[0][i][j] = floatArrayOf(
                        (Color.red(pixelval)).toFloat(),
                        (Color.green(pixelval)).toFloat(),
                        (Color.blue(pixelval)).toFloat()
                    )
                }
            }


//            val testBitmap =
//                Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888, true)
//            testBitmap.setPixels(testInput, 0, 128, 0, 0, 128, 128)
//
//            this.runOnUiThread {
//                binding.ivOutput.alpha = 0.8f
//                binding.ivOutput.setImageBitmap(
//                    processedBitmap//bitmap//processedBitmap
//                )
//            }


            when (modelStrategy) {
                strategyBackgroundRemover -> {
                    tflite?.run(inputBuffer, outputArray)
                    backgroundRemover(outputArray, inputBuffer)
                }

                strategyFreshnessClassifier -> {
                    val outArray = outputArray[0][0]
                    tflite?.run(inputBuffer, outArray)
                    freshnessClassifier(outArray)
                }
                else -> null
            }

            image.close()
        }


        cameraProvider.bindToLifecycle(
            this,
            cameraSelector,
            imageAnalysis,
            preview
        )
    }

    private fun backgroundRemover(
        outputArray: Array<Array<Array<FloatArray>>>,
        inputArray: Array<Array<Array<FloatArray>>>
    ) {
        val thisScaleW = 128
        val thisScaleH = 176
        val extraSpace = 0
        //arrayOf(Array(modelScaleH) { Array(modelScaleW) { floatArrayOf(1f) } })
        val withAlpha = IntArray((thisScaleH + extraSpace) * thisScaleW * 1) {
            Color.argb(
                (0.6f * 255).toInt(), 3,
                218, 197
            )
        }
        Log.d("FirstFragment", "running backgroundRemover")

//        val trimmedArray = arrayOf(Array(thisScaleH) { Array(thisScaleW) { floatArrayOf(1f) } })
//
//        for ((i, ix) in trimmedArray[0].withIndex()) {
//            for (j in ix.indices) {
//                trimmedArray[0][i][j] = outputArray[0][i][16 + j]//outputArray[0][i][12 + j]
//            }
//        }


        for ((i, ix) in outputArray[0].withIndex()) {
            for (j in ix.indices) {
                var alphaVal = 1f - outputArray[0][i][j][0]
                val cutoff = 0.6f
                alphaVal = if (alphaVal > cutoff) cutoff else alphaVal

                val pixelcolor = Color.argb(
                    (alphaVal * 255).toInt(), 3,
                    218, 197
                )

                withAlpha[(i + (extraSpace / 2)) * thisScaleW + j + 0] = pixelcolor

            }
        }
//            Log.d("FirstFragment", "withAlpha ${withAlpha.contentToString()}")
        val alphaBitmap =
            Bitmap.createBitmap(thisScaleW, thisScaleH + extraSpace, Bitmap.Config.ARGB_8888, true)
        alphaBitmap.setPixels(withAlpha, 0, thisScaleW, 0, 0, thisScaleW, thisScaleH + extraSpace)



        this.runOnUiThread {
            binding.ivOutput.setImageBitmap(
                alphaBitmap
            )
        }

    }


    private fun freshnessClassifier(outputArray: Array<FloatArray>) {//Array<FloatArray>
        //arrayOf(Array(modelScaleH) { Array(modelScaleW) { floatArrayOf(1f) } })
        Log.d("FirstFragment", "running freshnessClassifier")
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
        val maxValIndex = outputArray[0].indexOfFirst { it == outputArray[0].maxOrNull() }

        this.runOnUiThread {
            binding.tvResultOutput.text = labels[maxValIndex]
        }

    }


    private fun allPermissionsGranted() = PERMISSIONS_REQUIRED.all {
        ContextCompat.checkSelfPermission(
            this, it
        ) == PackageManager.PERMISSION_GRANTED
    }


    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        if (drawerToggle.onOptionsItemSelected(item)) {
            return true
        }


        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }


    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()

    }


}