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
import android.view.Surface.ROTATION_90
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

        // Navigation drawer side menu set up
        drawerToggle = ActionBarDrawerToggle(
            this, binding.drawerLayout,
            R.string.openDrawer,
            R.string.closeDrawer
        )
        binding.drawerLayout.addDrawerListener(drawerToggle)
        drawerToggle.syncState()

        // This was supposed to make it so that the back button closes the side menu, but
        // it doesn't work
//        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        binding.navViewMlList.setNavigationItemSelectedListener {

            when (it.itemId) {
                R.id.model_background_remover -> model = BackgroundRemover
                R.id.model_fresh_class -> {
                    // the close function here is to clear the output overlay of
                    // the BackgroundRemover, that's way it's called first
                    model.close(this, binding)
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
        //Set up camera preview
        val preview: Preview = Preview.Builder()
            .build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(binding.previewView.surfaceProvider)


        // Here we begin the tensorflow set up
        val imageAnalysis = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
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

            // This was a guessing game to make the output fit the camera preview
            // so it might not work properly on all devices.
            val imageProcessor = ImageProcessor.Builder()
                .add(Rot90Op(135))
//                .add(ResizeWithCropOrPadOp(sizeH, sizeW))
                .add(ResizeOp(modelScaleH, modelScaleW, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .build()

            // Create a TensorImage object. This creates the tensor of the corresponding
            // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs for
            // a quantized model soo the 'DataType' is UINT8 (8-bit unsigned integer)
            val tensorImage = TensorImage(DataType.UINT8)

            // Preprocess the image
            tensorImage.load(bitmap)

            val processedImage = imageProcessor.process(tensorImage)
            val processedBitmap = processedImage.bitmap

            // Initialize interpreter with NNAPI delegate for Android Pie or above
            // and just increase the number of threads otherwise
            // also this crashes the virtual device, not sure why.
            val tfliteOptions = Interpreter.Options()

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                Log.d("GPUFirstFragment", "Running on GPU")
                // DEPTHWISE_CONV doesn't run on GPU delagte
                // Although not sure it runs properly on NnApi either, but it does speed things up
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
                    "$modelString.tflite"
                )

                tflite = Interpreter(mappedByteBuffer, tfliteOptions)
            } catch (e: IOException) {
                Log.e("tfliteSupport", "Error reading model", e)
            } catch (e: IllegalStateException) {
                Log.e("tfliteSupport", "rotation err", e)
            }


            val inputBuffer =
                arrayOf(Array(modelScaleH) { Array(modelScaleW) { floatArrayOf(0f, 0f, 0f) } })

            val outputArray = model.outputArray


            // Convert the bitmap color values to an array of RGB values
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
            // Maybe I should train models that have the same input AND output shapes
            var result = arrayOf(Array(1) { Array(1) { FloatArray(1) } })


            // The ifs are checking if the model has finished switching as sometimes it
            // gives the output of the another model and it crashes
            when (model.fileName) {
                BackgroundRemover.fileName -> {
                    if (tflite?.getOutputTensor(0)?.shape()!![1] == 176) {
                        tflite.run(inputBuffer, outputArray)
                        result = outputArray
                    }
                }

                FreshnessClassifier.fileName -> {
                    if (tflite?.getOutputTensor(0)?.shape()!![1] == 12) {
                        val outArray = outputArray[0][0]
                        tflite.run(inputBuffer, outArray)
                        result = arrayOf(arrayOf(outArray))
                    }
                }
            }

            model.processOutput(this, binding, result)

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