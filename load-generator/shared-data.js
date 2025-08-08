import { SharedArray } from 'k6/data';
import encoding from 'k6/encoding';


// SharedArray for image data - this will be memory efficient across all VUs
export const sharedImageData = new SharedArray('imageData', function() {
    // Download image once and share across all VUs
    // Load image from local file "pic.jpg" instead of downloading
    const image = open("pic.jpg", "b");
    
    const imageDataB64 = encoding.b64encode(image);
    
    return [imageDataB64];
}); 