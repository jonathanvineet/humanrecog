import { NextRequest, NextResponse } from 'next/server';

const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const video = formData.get('video') as File;

    if (!video) {
      return NextResponse.json(
        { error: 'No video file provided' },
        { status: 400 }
      );
    }

    // Forward to Flask API
    const flaskFormData = new FormData();
    flaskFormData.append('video', video);

    const response = await fetch(`${FLASK_API_URL}/api/upload`, {
      method: 'POST',
      body: flaskFormData,
    });

    if (!response.ok) {
      throw new Error(`Flask API error: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Upload failed' },
      { status: 500 }
    );
  }
}
