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
import com.example.mlvstheworld.ModelInputDims.modelScaleH
import com.example.mlvstheworld.ModelInputDims.modelScaleW
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
    private var model: MLModelStats = BackgroundRemover
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
                R.id.model_background_remover -> model = BackgroundRemover
                R.id.model_fresh_class -> {
                    model.close(this,binding)
                    model = FreshnessClassifier
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


            // Initialization code
            // Create an ImageProcessor with all ops required. For more ops, please
            // refer to the ImageProcessor Architecture section in this README.

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
                val modelString = model.fileName

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

            val outputArray = model.outputArray


            val processedBitmap = processedImage.bitmap


            for ((i, ix) in inputBuffer[0].withIndex()) {
                for (j in ix.indices) {
                    val pixelval = processedBitmap.getPixel(j, i)

                    inputBuffer[0][i][j] = floatArrayOf(
                        (Color.red(pixelval)).toFloat(),
                        (Color.green(pixelval)).toFloat(),
                        (Color.blue(pixelval)).toFloat()
                    )
                }
            }


            // Not happy with this part
            // Maybe I should train models that have the same inputs and outputs
            var result = arrayOf(Array(1) { Array(1) { FloatArray(1) } })

            when (model.fileName) {
                BackgroundRemover.fileName -> {
                    tflite?.run(inputBuffer, outputArray)
                    result = outputArray
                }

                FreshnessClassifier.fileName -> {
                    val outArray = outputArray[0][0]
                    tflite?.run(inputBuffer, outArray)
                    result = arrayOf(arrayOf(outArray))
                }
                else -> null
            }

            model.processOutput(this,binding,result)

            image.close()
        }


        cameraProvider.bindToLifecycle(
            this,
            cameraSelector,
            imageAnalysis,
            preview
        )
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