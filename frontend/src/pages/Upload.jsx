import { useState, useCallback } from 'react';
import { uploadVideo } from '../services/api';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { HiCloudArrowUp, HiPlay, HiArrowRight, HiXMark, HiArrowLeft } from 'react-icons/hi2';
import { useProgress } from '../components/ProgressBar';
import StepNav from '../components/StepNav';
import './Upload.css';

export default function Upload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [uploadedVideoId, setUploadedVideoId] = useState(null);
    const navigate = useNavigate();
    const { start, done } = useProgress();

    const onDrop = useCallback(async (acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            const f = acceptedFiles[0];
            setFile(f);
            setPreview(URL.createObjectURL(f));

            try {
                start();
                const data = await uploadVideo(f, (progress) => {
                    // We could use a specific setProgress here if exposed, 
                    // but for now start/done handles the visual indeterminate state 
                    // or we can assume it finishes when done() is called.
                    // If ProgressBar supports value, we'd use it.
                });

                // Add a small delay for UX so user sees the "done" state
                setTimeout(() => {
                    done();
                    setUploadSuccess(true);
                    setUploadedVideoId(data.video_id);
                }, 800);
            } catch (error) {
                console.error("Upload failed", error);
                alert("Upload failed. Please try again.");
                done();
                setFile(null);
            }
        }
    }, [start, done, navigate]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'video/*': ['.mp4', '.mov', '.avi', '.mkv'] },
        maxFiles: 1,
    });

    const clearFile = () => {
        setFile(null);
        if (preview) URL.revokeObjectURL(preview);
        setPreview(null);
    };

    return (
        <div className="page-container upload-page">
            <div className="bg-grid" />




            <StepNav />

            <motion.div
                className="upload-page__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h1>Upload Match Footage</h1>
                <p>Drag &amp; drop your video file or click to browse</p>
            </motion.div>

            <AnimatePresence mode="wait">
                {!file ? (
                    <motion.div
                        key="dropzone"
                        {...getRootProps()}
                        className={`upload-zone ${isDragActive ? 'upload-zone--active' : ''}`}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.3 }}
                    >
                        <input {...getInputProps()} />
                        <motion.div
                            animate={{ y: [0, -8, 0] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                        >
                            <HiCloudArrowUp className="upload-zone__icon" />
                        </motion.div>
                        <p className="upload-zone__text">
                            {isDragActive ? 'Drop your video here...' : 'Drag video file here'}
                        </p>
                        <span className="upload-zone__hint">
                            Supports MP4, MOV, AVI, MKV — Max 2GB
                        </span>
                    </motion.div>
                ) : (
                    <motion.div
                        key="preview"
                        className="upload-preview"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <div className="upload-preview__video-wrap">
                            <video src={preview} controls className="upload-preview__video" />
                            <button className="upload-preview__clear" onClick={clearFile}>
                                <HiXMark />
                            </button>
                        </div>

                        <div className="upload-preview__info">
                            <div className="upload-preview__meta">
                                <HiPlay className="upload-preview__meta-icon" />
                                <div>
                                    <p className="upload-preview__filename">{file.name}</p>
                                    <p className="upload-preview__filesize">
                                        {(file.size / (1024 * 1024)).toFixed(1)} MB
                                    </p>
                                </div>
                            </div>
                            <button
                                className={`btn ${uploadSuccess ? 'btn-success' : 'btn-primary'}`}
                                onClick={() => {
                                    if (uploadSuccess && uploadedVideoId) {
                                        navigate('/trim', { state: { videoId: uploadedVideoId } });
                                    }
                                }}
                                disabled={!uploadSuccess}
                            >
                                {uploadSuccess ? 'Continue to Trim' : 'Uploading...'}
                                <HiArrowRight />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
