import { useState, useCallback } from 'react';
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
    const navigate = useNavigate();
    const { start, done } = useProgress();

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            start();
            const f = acceptedFiles[0];
            setFile(f);
            setPreview(URL.createObjectURL(f));
            setTimeout(done, 600);
        }
    }, [start, done]);

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

            <motion.div
                className="upload-page__back"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/')}>
                    <HiArrowLeft /> Back
                </button>
            </motion.div>

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
                                className="btn btn-primary"
                                onClick={() => navigate('/trim')}
                            >
                                Continue to Trim
                                <HiArrowRight />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
