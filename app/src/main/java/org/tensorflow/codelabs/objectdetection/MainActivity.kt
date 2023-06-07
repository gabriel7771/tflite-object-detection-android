/**
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.codelabs.objectdetection

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.util.SparseIntArray
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.google.gson.Gson
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

class MainActivity : AppCompatActivity(), View.OnClickListener {
    companion object {
        const val TAG = "TFLite - ODT"
        const val REQUEST_IMAGE_CAPTURE: Int = 1
        private const val MAX_FONT_SIZE = 96F
    }

    private lateinit var frameLayout: FrameLayout
    private lateinit var captureImageFab: Button
    private lateinit var addBoxButton: Button
    private lateinit var inputImageView: ImageView
    private lateinit var tvPlaceholder: TextView
    private lateinit var currentPhotoPath: String

    private val buttonsList = mutableListOf<BoxWithLabel>()

    private var scaleFactor = 1f
    private var originalImageWidth = 1
    private var originalImageHeight = 1

    private val options = BarcodeScannerOptions.Builder()
        .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
        .build()

    private val barcodeScanner = BarcodeScanning.getClient(options)

    private var qrPayload: QRPayload = QRPayload()

    private val gson = Gson()

    private var detectionResults = mutableListOf<DetectionResult>()

    private var pixelToCmRatio = 1f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        frameLayout = findViewById(R.id.frameLayout)
        captureImageFab = findViewById(R.id.captureImageFab)
        inputImageView = findViewById(R.id.imageView)
        tvPlaceholder = findViewById(R.id.tvPlaceholder)
        addBoxButton = findViewById(R.id.addBoxButton)

        captureImageFab.setOnClickListener(this)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE &&
            resultCode == Activity.RESULT_OK
        ) {
            val capturedBitmap = getCapturedImage()
            val barcodeImage = InputImage.fromBitmap(capturedBitmap, 0)
            barcodeScanner.process(barcodeImage)
                .addOnSuccessListener { barcodes ->
                    val barcodesPayload = barcodes?.mapNotNull {
                        try {
                            val boundingBox = it.boundingBox ?: return@mapNotNull null
                            val referenceQRPayload = gson.fromJson(it.rawValue, ReferenceQRPayload::class.java)
                            QRPayload(referenceQRPayload = referenceQRPayload, boundingBox = boundingBox)
                        } catch (e: Exception) {
                            null
                        }
                    }
                    if (barcodesPayload?.isNotEmpty() == true) {
                        qrPayload = barcodesPayload.first()
                    }
                    setViewAndDetect(capturedBitmap)
                    showAddBoxButton()
                }
                .addOnFailureListener {
                    Log.e(TAG, "onActivityResult: ", it)
                }
        }
    }

    private fun showAddBoxButton() {
        if (!addBoxButton.isVisible) {
            addBoxButton.setOnClickListener {
                addBox()
            }
        }
        addBoxButton.isVisible = inputImageView.drawable != null
    }

    private fun addBox() {
        val button = Button(this)
        Log.d(TAG, "runObjectDetection: scaleFactor: $scaleFactor")
        val width = 300
        val height = 300
        button.translationX = (frameLayout.width.toFloat() / 2f) - (width.toFloat() / 2f)
        button.translationY = (frameLayout.height.toFloat() / 2f) - (height.toFloat() / 2f)
        button.layoutParams = ViewGroup.LayoutParams(width, height)
        button.setBackgroundResource(R.drawable.shape_transparent_with_border)
        frameLayout.addView(button)
        val textView = TextView(this)
        textView.translationX = button.translationX
        textView.translationY = button.translationY
        textView.elevation = 2f
        setOnDragListener(button)
        frameLayout.addView(textView)
        buttonsList.add(BoxWithLabel(button, textView))
    }

    /**
     * onClick(v: View?)
     *      Detect touches on the UI components
     */
    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.captureImageFab -> {
                try {
                    dispatchTakePictureIntent()
                } catch (e: ActivityNotFoundException) {
                    Log.e(TAG, e.message.toString())
                }
            }
        }
    }

    /**
     * runObjectDetection(bitmap: Bitmap)
     *      TFLite Object Detection function
     */
    private fun runObjectDetection(bitmap: Bitmap) {
        // Step 1: Create TFLite's TensorImage object
        val image = TensorImage.fromBitmap(bitmap)

        // Step 2: Initialize the detector object
        val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(5)
                .setScoreThreshold(0.3f)
                .build()
        val detector = ObjectDetector.createFromFileAndOptions(
                this,
                "cabinets_model.tflite",
                options
        )

        // Step 3: Feed given image to the detector
        val results = detector.detect(image)

        // Step 4: Parse the detection result and show it
        val resultToDisplay = results.map {
            // Get the top-1 category and craft the display text
            val category = it.categories.first()
            val text = "${category.label}, ${category.score.times(100).toInt()}%"

            // Create a data object to display the detection result
            DetectionResult(it.boundingBox, text)
        }
        detectionResults.addAll(resultToDisplay)
        // Draw the detection result on the bitmap and show it.
        val imgWithResult = drawDetectionResult(bitmap, resultToDisplay)
        runOnUiThread {
            inputImageView.setImageBitmap(imgWithResult)
            for (button in buttonsList) {
                frameLayout.removeView(button.box)
                frameLayout.removeView(button.label)
            }
            val currentImageHeight = inputImageView.height
            scaleFactor = currentImageHeight.toFloat() / originalImageHeight.toFloat()
            Log.d(TAG, "runObjectDetection: qrPayload 1 ${qrPayload.boundingBox}")
            val qrWidthPx = (qrPayload.boundingBox.right - qrPayload.boundingBox.left) * scaleFactor
            Log.d(TAG, "runObjectDetection: qrWidthPx: $qrWidthPx")
            val qrPerimeterPx = 4 * qrWidthPx
            val qrWidthCm = qrPayload.referenceQRPayload.width
            val qrPerimeterCm = 4 * qrWidthCm
            pixelToCmRatio = qrPerimeterPx / qrPerimeterCm

//            for (result in resultToDisplay) {
//                val button = Button(this)
//                button.translationX = (result.boundingBox.centerX()) * scaleFactor
//                button.translationY = (result.boundingBox.top) * scaleFactor
//                Log.d(TAG, "runObjectDetection: scaleFactor: $scaleFactor")
//                val width = (result.boundingBox.right - result.boundingBox.left) * scaleFactor
//                val height = (result.boundingBox.bottom - result.boundingBox.top) * scaleFactor
//                button.layoutParams = ViewGroup.LayoutParams(width.toInt(), height.toInt())
//                button.setBackgroundResource(R.drawable.shape_transparent_with_border)
//                frameLayout.addView(button)
//                Log.d(TAG, "runObjectDetection: width: $width")
//                Log.d(TAG, "runObjectDetection: pixelToCmRatio: $pixelToCmRatio")
//                val widthCm = width / pixelToCmRatio
//                val heightCm = height / pixelToCmRatio
//                Log.d(TAG, "runObjectDetection: width cm: $widthCm height cm: $heightCm")
//                val textView = TextView(this)
//                val displayWidth = String.format("%.2f", widthCm)
//                val displayHeight = String.format("%.2f", heightCm)
//                textView.text = "${result.text} $displayWidth x $displayHeight cm"
//                textView.translationX = button.translationX
//                textView.translationY = button.translationY
//                setOnDragListener(button)
//                frameLayout.addView(textView)
//                buttonsList.add(BoxWithLabel(button, textView))
//            }



            /*
            val button = Button(this)
            buttonsList.add(button)
            button.translationX = (qrPayload.boundingBox.right) * scaleFactor
            button.translationY = (qrPayload.boundingBox.top) * scaleFactor
            Log.d(TAG, "runObjectDetection: qrPayload 2 ${qrPayload.boundingBox}")
            val width = (qrPayload.boundingBox.right - qrPayload.boundingBox.left) * scaleFactor
            val height = (qrPayload.boundingBox.bottom - qrPayload.boundingBox.top) * scaleFactor
            button.layoutParams = ViewGroup.LayoutParams(width.toInt(), height.toInt())
            button.setBackgroundResource(R.drawable.shape_transparent_with_border)
            frameLayout.addView(button)
             */
        }
    }

    private fun recalculateDimensions(button: BoxWithLabel) {
        val width = button.box.width
        val height = button.box.height
        val widthCm = width / pixelToCmRatio
        val heightCm = height / pixelToCmRatio
        val displayWidth = String.format("%.2f", widthCm)
        val displayHeight = String.format("%.2f", heightCm)
        button.label.text = "$displayWidth x $displayHeight cm"
        button.label.translationX = button.box.translationX
        button.label.translationY = button.box.translationY
    }

    private var startX = 0
    private var startY = 0
    private var touchX: TouchX = TouchX.LEFT
    private var touchY: TouchY = TouchY.TOP

    @SuppressLint("ClickableViewAccessibility")
    private fun setOnDragListener(button: Button) {
        button.setOnTouchListener { view, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    startX = event.x.toInt()
                    startY = event.y.toInt()
                    val centerX = (view.right + view.left) / 2
                    val centerY = (view.top + view.bottom) / 2
                    if (startX > centerX) {
                        touchX = TouchX.RIGHT
                    } else {
                        touchX = TouchX.LEFT
                    }
                    if (startY > centerY) {
                        touchY = TouchY.BOTTOM
                    } else {
                        touchY = TouchY.TOP
                    }
                }
                MotionEvent.ACTION_MOVE -> {
                    val endX = event.x.toInt()
                    val endY = event.y.toInt()
                    val params = view.layoutParams
                    val deltaDx = endX - startX
                    val deltaDy = endY - startY
                    when (touchX) {
                        TouchX.LEFT -> {
                            params.width -= deltaDx
                            view.translationX += deltaDx
                        }
                        TouchX.RIGHT -> {
                            params.width += deltaDx
                        }
                    }
                    when (touchY) {
                        TouchY.BOTTOM -> {
                            params.height += deltaDy
                        }
                        TouchY.TOP -> {
                            params.height -= deltaDy
                            view.translationY += deltaDy
                        }
                    }
                    startX = event.x.toInt()
                    startY = event.y.toInt()
                    val boxWithLabel = buttonsList.find { it.box == button }
                    boxWithLabel?.let {
                        recalculateDimensions(boxWithLabel)
                    }
                    view.requestLayout()
                }
            }
            true
        }
    }

    /**
     * debugPrint(visionObjects: List<Detection>)
     *      Print the detection result to logcat to examine
     */
    private fun debugPrint(results : List<Detection>) {
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox

            Log.d(TAG, "Detected object: $i ")
            Log.d(TAG, "  boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")

            for ((j, category) in obj.categories.withIndex()) {
                Log.d(TAG, "    Label $j: ${category.label}")
                val confidence: Int = category.score.times(100).toInt()
                Log.d(TAG, "    Confidence: ${confidence}%")
            }
        }
    }

    /**
     * setViewAndDetect(bitmap: Bitmap)
     *      Set image to view and call object detection
     */
    private fun setViewAndDetect(bitmap: Bitmap) {
        // Display capture image
        inputImageView.setImageBitmap(bitmap)
        tvPlaceholder.visibility = View.INVISIBLE

        // Run ODT and display result
        // Note that we run this in the background thread to avoid blocking the app UI because
        // TFLite object detection is a synchronised process.
        lifecycleScope.launch(Dispatchers.Default) { runObjectDetection(bitmap) }
    }

    /**
     * getCapturedImage():
     *      Decodes and crops the captured image from camera.
     */
    private fun getCapturedImage(): Bitmap {
        // Get the dimensions of the View
        val targetW: Int = inputImageView.width
        val targetH: Int = inputImageView.height

        val bmOptions = BitmapFactory.Options().apply {
            // Get the dimensions of the bitmap
            inJustDecodeBounds = true

            BitmapFactory.decodeFile(currentPhotoPath, this)

            val photoW: Int = outWidth
            val photoH: Int = outHeight

            // Determine how much to scale down the image
            val scaleFactor: Int = max(1, min(photoW / targetW, photoH / targetH))
            Log.d(TAG, "getCapturedImage: scaleFactor: $scaleFactor")

            // Decode the image file into a Bitmap sized to fill the View
            inJustDecodeBounds = false
            inSampleSize = scaleFactor
            inMutable = true
        }
        val exifInterface = ExifInterface(currentPhotoPath)
        val orientation = exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )

        val bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                rotateImage(bitmap, 90f)
            }
            ExifInterface.ORIENTATION_ROTATE_180 -> {
                rotateImage(bitmap, 180f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> {
                rotateImage(bitmap, 270f)
            }
            else -> {
                bitmap
            }
        }
    }

    /**
     * getSampleImage():
     *      Get image form drawable and convert to bitmap.
     */
    private fun getSampleImage(drawable: Int): Bitmap {
        return BitmapFactory.decodeResource(resources, drawable, BitmapFactory.Options().apply {
            inMutable = true
        })
    }

    /**
     * rotateImage():
     *     Decodes and crops the captured image from camera.
     */
    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    /**
     * createImageFile():
     *     Generates a temporary image file for the Camera app to write to.
     */
    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    /**
     * dispatchTakePictureIntent():
     *     Start the Camera app to take a photo.
     */
    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (e: IOException) {
                    Log.e(TAG, e.message.toString())
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "org.tensorflow.codelabs.objectdetection.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    /**
     * drawDetectionResult(bitmap: Bitmap, detectionResults: List<DetectionResult>
     *      Draw a box around each objects and show the object's name.
     */
    private fun drawDetectionResult(
        bitmap: Bitmap,
        detectionResults: List<DetectionResult>
    ): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT
        Log.d(TAG, "drawDetectionResult: canvas width: ${canvas.width}")
        Log.d(TAG, "drawDetectionResult: canvas height: ${canvas.height}")
        Log.d(TAG, "drawDetectionResult: detectionResults: $detectionResults")
        originalImageWidth = canvas.width
        originalImageHeight = canvas.height
        detectionResults.forEach {
            // draw bounding box
            pen.color = Color.RED
            pen.strokeWidth = 8F
            pen.style = Paint.Style.STROKE
            val box = it.boundingBox
            //canvas.drawRect(box, pen)

            val tagSize = Rect(0, 0, 0, 0)

            // calculate the right font size
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.YELLOW
            pen.strokeWidth = 2F

            pen.textSize = MAX_FONT_SIZE
            pen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()

            // adjust the font size so texts are inside the bounding box
            if (fontSize < pen.textSize) pen.textSize = fontSize

            var margin = (box.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F
            //canvas.drawText(it.text, box.left + margin, box.top + tagSize.height().times(1F), pen)
        }
        return outputBitmap
    }
}

/**
 * DetectionResult
 *      A class to store the visualization info of a detected object.
 */
data class DetectionResult(val boundingBox: RectF, val text: String)

data class ReferenceQRPayload(
    val width: Float = 0f,
    val height: Float = 0f,
    val units: String = "cm"
)

data class QRPayload(
    val referenceQRPayload: ReferenceQRPayload = ReferenceQRPayload(),
    val boundingBox: Rect = Rect()
)

data class BoxWithLabel(
    val box: Button,
    val label: TextView
)

enum class TouchX {
    LEFT,
    RIGHT
}

enum class TouchY {
    TOP,
    BOTTOM
}
