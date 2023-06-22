//
//  ViewController.swift
//  CoinDetector
//
//  Created by M Alfin Syahruddin on 17/06/23.
//

import UIKit
import AVFoundation
import Vision
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var totalLabel: UILabel!
    
    private var request: VNCoreMLRequest!
    
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var detectionLayer: CALayer!

    private let session = AVCaptureSession()
    private var cameraSize: CGSize!
    private let videoOutputQueue = DispatchQueue(label: "video-output-queue", qos: .userInitiated)

    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupVision()
        setupCaptureSession()
        
        setupLayers()
        updateLayerGeometry()
        
        DispatchQueue.global(qos: .background).async {
            self.session.startRunning()
        }
    }

    
    private func setupVision() {
        guard let coinDetector = try? CoinDetector(configuration: MLModelConfiguration()) else {
            fatalError("Failed to create an object detector model instance.")
        }
        
        guard let model = try? VNCoreMLModel(for: coinDetector.model) else {
            fatalError("Failed to create a `VNCoreMLModel` instance.")
        }
        
        let request = VNCoreMLRequest(
            model: model,
            completionHandler: visionRequestHandler
        )
        request.imageCropAndScaleOption = .scaleFit
        self.request = request
    }
    
    private func setupCaptureSession() {
        session.beginConfiguration()
        
        // Add the video input to the capture session
        let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        )!
  
        // Connect the camera to the capture session input
        let cameraInput = try! AVCaptureDeviceInput(device: camera)
        session.addInput(cameraInput)

        session.sessionPreset = .vga640x480

        // Create the video data output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            String(kCVPixelBufferPixelFormatTypeKey): Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
        
        // Add the video output to the capture session
        session.addOutput(videoOutput)
        
        // Set camera size
        let dimension = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
        cameraSize = CGSize(
            width: CGFloat(dimension.width),
            height: CGFloat(dimension.height)
        )
        
        session.commitConfiguration()
    }
    
    private func setupLayers() {
        // Configure the preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = self.previewView.layer.bounds
        self.previewView.layer.addSublayer(previewLayer)
        
        // Configure the detection Layer
        detectionLayer = CALayer()
        detectionLayer.bounds = CGRect(
            x: 0.0,
            y: 0.0,
            width: cameraSize.width,
            height: cameraSize.height
        )
        detectionLayer.position = CGPoint(
            x: self.previewView.layer.bounds.midX,
            y: self.previewView.layer.bounds.midY
        )
        self.previewView.layer.addSublayer(detectionLayer)
    }
    
    private func updateLayerGeometry() {
        let bounds = self.previewView.layer.bounds

        let xScale: CGFloat = bounds.size.width / cameraSize.height
        let yScale: CGFloat = bounds.size.height / cameraSize.width

        var scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
        // rotate the layer into screen orientation, scale and mirror
        detectionLayer.setAffineTransform(
            CGAffineTransform(
                rotationAngle: CGFloat(.pi / 2.0)
            )
            .scaledBy(x: scale, y: -scale)
        )

        // center the layer
        detectionLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)

        CATransaction.commit()
    }
    
    private func visionRequestHandler(_ request: VNRequest, error: Error?) {
        if let error = error {
            print("Vision image detection error: \(error.localizedDescription)")
            return
        }

        if request.results == nil {
            print("Vision request had no results.")
            return
        }

        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            print("VNRequest produced the wrong result type: \(type(of: request.results)).")
            return
        }
                
        DispatchQueue.main.async {
            CATransaction.begin()
            CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
            self.detectionLayer.sublayers = nil // remove all the old recognized objects
            for observation in observations {
                let objectBounds = VNImageRectForNormalizedRect(
                    observation.boundingBox,
                    Int(self.cameraSize.width),
                    Int(self.cameraSize.height)
                )
                                
                let shapeLayer = self.createRoundedRectLayer(objectBounds)
                self.detectionLayer.addSublayer(shapeLayer)
            }
            self.updateLayerGeometry()
            CATransaction.commit()
            
            // Set Total Label
            let total = observations.count
            self.totalLabel.text = "\(total) Coins"
        }
    }
    
    private func createRoundedRectLayer(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.backgroundColor = UIColor.yellow.withAlphaComponent(0.15).cgColor
        shapeLayer.cornerRadius = 8
        shapeLayer.borderColor = UIColor.yellow.cgColor
        shapeLayer.borderWidth = 1.5
        return shapeLayer
    }
}


extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        do {
            let handler = VNImageRequestHandler(cvPixelBuffer: imageBuffer)
            try handler.perform([request])
        } catch {
            print(error.localizedDescription)
        }
    }
}

