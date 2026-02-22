import { NextRequest, NextResponse } from 'next/server';

const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function GET(
  request: NextRequest,
  { params }: { params: { taskId: string } }
) {
  try {
    const { taskId } = params;

    if (!taskId) {
      return NextResponse.json(
        { error: 'Task ID is required' },
        { status: 400 }
      );
    }

    // Forward to Flask API
    const response = await fetch(`${FLASK_API_URL}/api/status/${taskId}`);

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Task not found' },
        { status: 404 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Status check error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Status check failed' },
      { status: 500 }
    );
  }
}
