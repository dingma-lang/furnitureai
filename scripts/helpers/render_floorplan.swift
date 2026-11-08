// PDFKit renders one bounded page; document scripts/attachments are never executed.
import AppKit
import PDFKit
let args = CommandLine.arguments
if args.count != 4 { exit(2) }
guard let doc = PDFDocument(url: URL(fileURLWithPath: args[1])), !doc.isLocked, doc.pageCount > 0, doc.pageCount <= 30, let index = Int(args[2]), index >= 0, index < doc.pageCount, let page = doc.page(at: index) else { fputs("Unreadable, locked, excessive or missing PDF page\n", stderr); exit(2) }
let bounds = page.bounds(for: .mediaBox)
guard bounds.width.isFinite, bounds.height.isFinite, bounds.width > 0, bounds.height > 0 else { exit(2) }
let scale = min(1600/bounds.width, 1600/bounds.height)
let size = NSSize(width: bounds.width*scale, height: bounds.height*scale)
let image = page.thumbnail(of: size, for: .mediaBox)
guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data:tiff), let png = bitmap.representation(using:.png,properties:[:]) else { exit(2) }
try png.write(to: URL(fileURLWithPath:args[3]))
print(doc.pageCount)
