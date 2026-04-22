import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    HiCloudArrowUp,
    HiPlay,
    HiArrowRight,
    HiXMark,
} from 'react-icons/hi2';
import { uploadVideo } from '../services/api';
import { useProgress } from '../components/ProgressBar';
import StepNav from '../components/StepNav';
import './Upload.css';

export default function Upload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [uploadPct, setUploadPct] = useState(0);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [uploadedVideoId, setUploadedVideoId] = useState(null);
    const navigate = useNavigate();
    const { start, done } = useProgress();

    const onDrop = useCallback(async (acceptedFiles) => {
        if (!acceptedFiles.length) return;
        const f = acceptedFiles[0];
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setUploadPct(0);
        setUploadSuccess(false);

        const toastId = toast.loading('Uploading video…');
        try {
            start();
            const data = await uploadVideo(f, (pct) => {
                setUploadPct(pct);
                toast.loading(`Uploading video… ${pct}%`, { id: toastId });
            });
            done();
            setUploadSuccess(true);
            setUploadedVideoId(data.session_id || data.video_id);
            toast.success('Upload complete', { id: toastId });
        } catch (err) {
            console.error('Upload failed', err);
            done();
            setFile(null);
            setPreview(null);
            toast.error(
                err?.response?.data?.detail ||
                    err?.message ||
                    'Upload failed. Check the server connection and try again.',
                { id: toastId }
            );
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
        setUploadSuccess(false);
        setUploadPct(0);
    };

    const goToTrim = () => {
        if (!uploadSuccess || !uploadedVideoId) return;
        navigate('/trim', {
            state: { videoId: uploadedVideoId, sessionId: uploadedVideoId },
        });
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
                <p>Drag &amp; drop your video or click to browse</p>
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
                            {isDragActive ? 'Drop your video here…' : 'Drag video file here'}
                        </p>
                        <span className="upload-zone__hint">
                            Supports MP4, MOV, AVI, MKV — chunked upload, up to 2 GB
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
                            <button
                                className="upload-preview__clear"
                                onClick={clearFile}
                                disabled={!uploadSuccess && uploadPct > 0 && uploadPct < 100}
                                title="Remove"
                            >
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
                                        {!uploadSuccess && uploadPct > 0 && ` · ${uploadPct}%`}
                                    </p>
                                </div>
                            </div>
                            <button
                                className={`btn ${uploadSuccess ? 'btn-success' : 'btn-primary'}`}
                                onClick={goToTrim}
                                disabled={!uploadSuccess}
                            >
                                {uploadSuccess ? 'Continue' : `Uploading… ${uploadPct}%`}
                                <HiArrowRight />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
