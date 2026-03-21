# 🖐️ Enhanced Hand Gesture Recognition System

Transform your computer into a touch-free control center using advanced hand gesture recognition technology.

## 🚀 Features

### Core Capabilities
- **Real-time hand detection** using MediaPipe
- **Multi-hand support** (up to 2 hands simultaneously)
- **21-point hand landmark tracking**
- **Smooth gesture recognition** with configurable sensitivity

### Practical Applications
- **Virtual Mouse Control** - Point gesture moves cursor
- **Click & Drag** - Pinch gesture for clicking
- **Presentation Remote** - Navigate slides hands-free
- **Volume Control** - Gesture-based volume adjustment
- **Media Control** - Play/pause with hand gestures
- **Accessibility Aid** - Touch-free computer interaction

### Supported Gestures
- **Point** - Virtual mouse control
- **Pinch** - Click action
- **Peace ✌️** - Next slide/track
- **Call Me 🤙** - Previous slide/track
- **Thumb Up** - Volume up
- **Two Fingers** - Volume down
- **Palm** - Pause/Play
- **Fist** - Release drag
- **Rock 🤘** - Custom action

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows/macOS/Linux

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the enhanced version
python enhanced_gesture_control.py

# Or run the original simple version
python main.py
```

## 📖 Usage

1. **Start the application**
   ```bash
   python enhanced_gesture_control.py
   ```

2. **Allow camera access** when prompted

3. **Position your hand** in view of the webcam

4. **Use gestures** to control your computer:
   - Point with index finger to move mouse
   - Make pinch (thumb + index) to click
   - Show peace sign for next slide
   - Show "call me" sign for previous slide
   - Thumbs up for volume up
   - Two fingers for volume down
   - Open palm for pause/play

5. **Press 'q'** to quit the application

## ⚙️ Configuration

Edit `config.json` to customize:
- Detection sensitivity
- Gesture hold time
- Mouse sensitivity
- Volume control steps
- UI appearance
- Data logging options

## 📊 Features

### Advanced UI
- Real-time gesture display
- Hand landmark visualization
- Control instructions overlay
- Session statistics
- Performance metrics

### Data Tracking
- Gesture usage statistics
- Session duration tracking
- JSON export of session data
- Performance analytics

### Accessibility
- High contrast mode option
- Large text support
- Audio feedback (configurable)
- Customizable gesture mapping

## 🔧 Technical Details

- **MediaPipe** for hand tracking
- **OpenCV** for camera processing
- **PyAutoGUI** for system control
- **NumPy** for calculations
- **JSON** for configuration and data

## 🌐 Real-World Applications

### Presentations
- Navigate PowerPoint/Google Slides
- Control video playback
- Laser pointer replacement

### Accessibility
- Assist users with mobility limitations
- Touch-free computer operation
- Alternative input method

### Creative
- Digital art control
- Music production
- Gaming accessibility

### Professional
- Clean room operations
- Medical applications
- Food industry use

## 📝 Requirements.txt
```
mediapipe>=0.10.0
opencv-python>=4.8.0
pyautogui>=0.9.54
numpy>=1.24.0
```

## 🔒 Privacy & Security

- All processing done locally
- No data sent to external servers
- Camera access only during runtime
- Optional session data logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

**Camera not working?**
- Check camera permissions
- Ensure no other app is using the camera
- Try different camera index (0, 1, 2...)

**Gestures not recognized?**
- Ensure good lighting
- Keep hand clearly visible
- Adjust detection confidence in config.json

**Mouse control not working?**
- Check pyautogui installation
- Ensure no security software blocking it
- Run as administrator if needed

## 📞 Support

For issues and feature requests, please open an issue on the project repository.

---

**Transform the way you interact with technology - one gesture at a time!** 🚀
