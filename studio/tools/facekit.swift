// facekit: face detection, landmarks, capture quality and image feature prints via Apple Vision.
// Built into macOS; no extra software. Usage:
//   facekit <image> [<image> ...]        -> one JSON object per line on stdout
//   facekit --segment <in> <out.png>     -> the person on a plain light-grey background (Vision person
//                                           segmentation); prints {"coverage": fraction of pixels kept}
// Each object: {"file", "width", "height", "faces": [{"confidence", "bbox": [x,y,w,h] (pixels, top-left origin),
//   "quality", "yaw", "roll", "pitch", "inner_lips": [[x,y]...], "outer_lips": [...], "face_contour": [...],
//   "left_eye": [...], "right_eye": [...], "nose": [...], "feature": [floats]}]}
// "feature" is a Vision image feature print of the square face crop (a generic image descriptor,
// NOT a face-recognition embedding).
import Foundation
import Vision
import CoreImage
import ImageIO

func cgImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func points(_ region: VNFaceLandmarkRegion2D?, _ w: Int, _ h: Int) -> [[Double]] {
    guard let r = region else { return [] }
    return r.pointsInImage(imageSize: CGSize(width: w, height: h)).map { [Double($0.x), Double(h) - Double($0.y)] }
}

func featurePrint(_ img: CGImage, crop: CGRect) -> [Float]? {
    guard let cropped = img.cropping(to: crop) else { return nil }
    let req = VNGenerateImageFeaturePrintRequest()
    let handler = VNImageRequestHandler(cgImage: cropped, options: [:])
    do { try handler.perform([req]) } catch { return nil }
    guard let obs = req.results?.first as? VNFeaturePrintObservation, obs.elementType == .float else { return nil }
    return obs.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func segment(_ input: String, _ output: String) -> Int32 {
    guard let img = cgImage(input) else { emit(["error": "unreadable image"]); return 1 }
    let req = VNGeneratePersonSegmentationRequest()
    req.qualityLevel = .accurate
    req.outputPixelFormat = kCVPixelFormatType_OneComponent8
    do { try VNImageRequestHandler(cgImage: img, options: [:]).perform([req]) } catch { emit(["error": "\(error)"]); return 1 }
    guard let mask = req.results?.first?.pixelBuffer else { emit(["error": "no person found"]); return 1 }
    let source = CIImage(cgImage: img)
    var maskImage = CIImage(cvPixelBuffer: mask)
    maskImage = maskImage.transformed(by: CGAffineTransform(scaleX: source.extent.width / maskImage.extent.width,
                                                            y: source.extent.height / maskImage.extent.height))
    let grey = CIImage(color: CIColor(red: 0.86, green: 0.87, blue: 0.87)).cropped(to: source.extent)
    let blend = CIFilter(name: "CIBlendWithMask", parameters: [kCIInputImageKey: source, kCIInputBackgroundImageKey: grey,
                                                               kCIInputMaskImageKey: maskImage])!
    let ctx = CIContext()
    guard let out = blend.outputImage, let cg = ctx.createCGImage(out, from: source.extent),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: output) as CFURL, "public.png" as CFString, 1, nil)
    else { emit(["error": "could not write output"]); return 1 }
    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
    // Coverage: mean mask value.
    CVPixelBufferLockBaseAddress(mask, .readOnly)
    let w = CVPixelBufferGetWidth(mask), h = CVPixelBufferGetHeight(mask), row = CVPixelBufferGetBytesPerRow(mask)
    let base = CVPixelBufferGetBaseAddress(mask)!.assumingMemoryBound(to: UInt8.self)
    var total = 0
    for y in 0..<h { for x in 0..<w { total += Int(base[y * row + x]) } }
    CVPixelBufferUnlockBaseAddress(mask, .readOnly)
    emit(["coverage": Double(total) / Double(w * h * 255)])
    return 0
}

let args = Array(CommandLine.arguments.dropFirst())
if args.first == "--segment" {
    guard args.count == 3 else { emit(["error": "usage: facekit --segment <in> <out.png>"]); exit(2) }
    exit(segment(args[1], args[2]))
}

for path in args {
    var out: [String: Any] = ["file": path]
    guard let img = cgImage(path) else { out["error"] = "unreadable image"; emit(out); continue }
    let w = img.width, h = img.height
    out["width"] = w; out["height"] = h
    let rects = VNDetectFaceRectanglesRequest()
    let marks = VNDetectFaceLandmarksRequest()
    let quality = VNDetectFaceCaptureQualityRequest()
    let handler = VNImageRequestHandler(cgImage: img, options: [:])
    do { try handler.perform([rects, marks, quality]) } catch { out["error"] = "\(error)"; emit(out); continue }
    let qualities = (quality.results ?? []).map { ($0.boundingBox, $0.faceCaptureQuality ?? -1) }
    var faces: [[String: Any]] = []
    for f in marks.results ?? [] {
        let b = f.boundingBox
        let bbox = [Double(b.minX) * Double(w), (1 - Double(b.maxY)) * Double(h), Double(b.width) * Double(w), Double(b.height) * Double(h)]
        var face: [String: Any] = ["confidence": Double(f.confidence), "bbox": bbox]
        face["quality"] = Double(qualities.min(by: { abs($0.0.midX - b.midX) + abs($0.0.midY - b.midY) < abs($1.0.midX - b.midX) + abs($1.0.midY - b.midY) })?.1 ?? -1)
        face["yaw"] = f.yaw?.doubleValue ?? 0; face["roll"] = f.roll?.doubleValue ?? 0; face["pitch"] = f.pitch?.doubleValue ?? 0
        let l = f.landmarks
        face["inner_lips"] = points(l?.innerLips, w, h); face["outer_lips"] = points(l?.outerLips, w, h)
        face["face_contour"] = points(l?.faceContour, w, h); face["left_eye"] = points(l?.leftEye, w, h)
        face["right_eye"] = points(l?.rightEye, w, h); face["nose"] = points(l?.nose, w, h)
        // Square crop centred on the face, 1.3x its size, clamped to the image.
        let side = max(bbox[2], bbox[3]) * 1.3
        let cx = bbox[0] + bbox[2] / 2, cy = bbox[1] + bbox[3] / 2
        let crop = CGRect(x: max(0, cx - side / 2), y: max(0, cy - side / 2), width: min(side, Double(w)), height: min(side, Double(h)))
        if let fp = featurePrint(img, crop: crop) { face["feature"] = fp }
        faces.append(face)
    }
    out["faces"] = faces
    emit(out)
}

func emit(_ o: [String: Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: o), let s = String(data: d, encoding: .utf8) { print(s) }
}
