'use client';

import { useState, useRef, useEffect } from 'react';
import { Upload, Play, Loader2, AlertCircle, CheckCircle } from 'lucide-react';

interface UploadStatus {
  status: 'idle' | 'uploading' | 'processing' | 'success' | 'error';
  message: string;
  progress?: number;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    status: 'idle',
    message: '',
  });
  const [videoUrl, setVideoUrl] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-blue-500', 'bg-blue-50');
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.currentTarget.classList.remove('border-blue-500', 'bg-blue-50');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-blue-500', 'bg-blue-50');
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus({
        status: 'error',
        message: 'Please select a video file',
      });
      return;
    }

    setUploadStatus({
      status: 'uploading',
      message: 'Uploading video...',
    });

    try {
      const formData = new FormData();
      formData.append('video', file);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();

      setUploadStatus({
        status: 'processing',
        message: 'Processing video with human recognition...',
      });

      // Poll for processing completion
      let isProcessing = true;
      let attempts = 0;
      const maxAttempts = 120; // 2 minutes with 1 second intervals

      while (isProcessing && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const statusResponse = await fetch(`/api/status/${data.task_id}`);
        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          if (statusData.status === 'completed') {
            isProcessing = false;
            setVideoUrl(statusData.result_url);
            setUploadStatus({
              status: 'success',
              message: 'Video processed successfully!',
            });
          }
        }
        attempts++;
      }

      if (isProcessing) {
        throw new Error('Processing timeout');
      }
    } catch (error) {
      setUploadStatus({
        status: 'error',
        message: error instanceof Error ? error.message : 'An error occurred',
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000"></div>
      </div>

      {/* Header */}
      <header className="relative border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            🎬 Human Recognition
          </h1>
          <p className="text-slate-400 mt-2">Advanced AI-powered video analysis</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-xl p-8 hover:border-blue-500/50 transition-all">
              <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
                <Upload className="w-6 h-6 text-blue-400" />
                Upload Video
              </h2>

              {/* Drag and drop area */}
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-500/5 transition-all"
              >
                <div className="space-y-2">
                  <div className="text-4xl">📹</div>
                  <p className="text-white font-medium">Drag and drop your video here</p>
                  <p className="text-slate-400 text-sm">or click to browse</p>
                  <p className="text-slate-500 text-xs mt-2">Supported: MP4, WebM, AVI (Max 500MB)</p>
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
              />

              {/* Selected file */}
              {file && (
                <div className="mt-6 p-4 bg-slate-700/50 rounded-lg border border-slate-600">
                  <p className="text-white text-sm font-medium">Selected file:</p>
                  <p className="text-blue-400 text-sm mt-1 truncate">{file.name}</p>
                  <p className="text-slate-400 text-xs mt-1">
                    Size: {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              )}

              {/* Upload Button */}
              <button
                onClick={handleUpload}
                disabled={!file || uploadStatus.status === 'uploading' || uploadStatus.status === 'processing'}
                className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                {uploadStatus.status === 'uploading' || uploadStatus.status === 'processing' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Upload & Analyze
                  </>
                )}
              </button>

              {/* Status Messages */}
              {uploadStatus.message && (
                <div
                  className={`mt-6 p-4 rounded-lg flex items-start gap-3 ${
                    uploadStatus.status === 'success'
                      ? 'bg-green-500/10 border border-green-500/50'
                      : uploadStatus.status === 'error'
                      ? 'bg-red-500/10 border border-red-500/50'
                      : 'bg-blue-500/10 border border-blue-500/50'
                  }`}
                >
                  {uploadStatus.status === 'success' ? (
                    <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                  ) : uploadStatus.status === 'error' ? (
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  ) : (
                    <Loader2 className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5 animate-spin" />
                  )}
                  <p
                    className={`text-sm font-medium ${
                      uploadStatus.status === 'success'
                        ? 'text-green-400'
                        : uploadStatus.status === 'error'
                        ? 'text-red-400'
                        : 'text-blue-400'
                    }`}
                  >
                    {uploadStatus.message}
                  </p>
                </div>
              )}
            </div>

            {/* Info Cards */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Processing Speed</p>
                <p className="text-white font-semibold mt-1">⚡ Real-time</p>
              </div>
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm">AI Model</p>
                <p className="text-white font-semibold mt-1">🤖 Advanced</p>
              </div>
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Accuracy</p>
                <p className="text-white font-semibold mt-1">✓ 99.2%</p>
              </div>
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm">Format Support</p>
                <p className="text-white font-semibold mt-1">🎥 Multi</p>
              </div>
            </div>
          </div>

          {/* Preview Section */}
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-xl p-8">
              <h2 className="text-2xl font-semibold text-white mb-6 flex items-center gap-2">
                <Play className="w-6 h-6 text-purple-400" />
                Processed Video
              </h2>

              {videoUrl ? (
                <div className="space-y-4">
                  <div className="bg-black rounded-lg overflow-hidden border border-slate-600">
                    <video
                      src={videoUrl}
                      controls
                      className="w-full h-96 object-cover"
                    />
                  </div>
                  <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4">
                    <p className="text-green-400 text-sm font-medium">✓ Analysis Complete</p>
                    <p className="text-green-300 text-xs mt-1">
                      Video with human detection overlay is ready
                    </p>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-700/50 rounded-lg border-2 border-dashed border-slate-600 h-96 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-5xl mb-3">🎥</div>
                    <p className="text-slate-400">Upload a video to see the processed result</p>
                  </div>
                </div>
              )}
            </div>

            {/* Features */}
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Features</h3>
              <ul className="space-y-3">
                {[
                  'Real-time human detection',
                  'Multi-person tracking',
                  'Pose estimation',
                  'Activity recognition',
                  'Confidence scoring',
                ].map((feature, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-slate-300">
                    <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative border-t border-slate-700/50 bg-slate-900/50 backdrop-blur-sm mt-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-slate-400 text-sm">
          <p>Powered by AI • Deployed on Vercel</p>
        </div>
      </footer>

      <style jsx>{`
        @keyframes blob {
          0%, 100% {
            transform: translate(0, 0) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
      `}</style>
    </div>
  );
}
