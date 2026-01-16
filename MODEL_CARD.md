# Ancient Manuscript Document Digitizer

## Model Description
A CRNN (Convolutional Recurrent Neural Network) model for recognizing text in ancient manuscripts.

**Architecture**: CNN Feature Extractor + Bidirectional LSTM + CTC Loss  
**Framework**: PyTorch  
**Task**: Optical Character Recognition (OCR)

## Performance Metrics
- **Character Error Rate (CER)**: 0.61%
- **Word Error Rate (WER)**: 1.51%
- **Exact Match Accuracy**: 98.49%
- **Inference Time**: 6.44ms per image
- **Throughput**: 155.39 images/second

## Model Details
- **Parameters**: 10,808,591
- **Input Size**: 64 x 256 (H x W)
- **Vocabulary Size**: 15 characters
- **Training Dataset**: AncientScriptNet (246,658 images)

## Training
- **Epochs**: 12
- **Best Epoch**: 9
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CTC Loss
- **Early Stopping**: Yes (patience=7)

## Usage
```python
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = T.Compose([
    T.Resize((64, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

image = Image.open('manuscript.jpg').convert('L')
image = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(image)
    prediction = ctc_decode(output, idx_to_char)[0]

print(f"Recognized text: {prediction}")
```

## Files
- `best_model.pth`: Best model checkpoint
- `final_model_complete.pth`: Complete model with all metadata
- `training_history.json`: Training metrics over epochs
- `test_results.json`: Final test set evaluation
- `category_metrics.json`: Per-category performance

## Citation
```
@misc{ancient-manuscript-ocr,
  title={Ancient Manuscript Document Digitizer},
  author={Shubham Kumar},
  year={2026},
  note={CRNN model for OCR on ancient manuscripts}
}
```

## Contact
Shubham Kumar] - [shubham24101@iiitnr.edu.in]  
Project Link: [https://github.com/CosmicShubham1/ancient-manuscript-ocr]
