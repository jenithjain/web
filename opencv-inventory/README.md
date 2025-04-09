# OpenCV Inventory Tracking System

This system uses OpenCV and Google's Gemini AI to track inventory movements in real-time through your webcam.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python inventory_tracker.py
```

## Usage

- The system will automatically start your webcam and begin monitoring for movement
- When movement is detected, it will capture the frame and analyze it using Gemini AI
- Press 'q' to quit the application

## Features

- Real-time motion detection
- AI-powered scene analysis using Google's Gemini
- Live video feed with analysis results 