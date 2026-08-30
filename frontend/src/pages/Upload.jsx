import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiCloudArrowUp, HiPlay, HiArrowRight, HiXMark } from 'react-icons/hi2';
import { uploadVideo } from '../services/api';
import { compressVideoIfNeeded } from '../utils/compressVideo';
import { useProgress } from '../components/ProgressBar';
import StepNav from '../components/StepNav';
import './Upload.css';

export default function Upload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [uploadPct, setUploadPct] = useState(0);
    const [isCompressing, setIsCompressing] = useState(false);
    const [compressPct, setCompressPct] = useState(0);
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
        setCompressPct(0);
        setUploadSuccess(false);

        let finalFile = f;
        start();

        // 1. 本地压缩预处理 (拦截长视频)
        if (f.size > 500 * 1024 * 1024) {
            setIsCompressing(true);
            const compressToastId = toast.loading('正在为您进行 AI 预处理...');
            try {
                finalFile = await compressVideoIfNeeded(f, (pct) => {
                    setCompressPct(pct);
                    toast.loading(`正在为您进行 AI 预处理... ${pct}%`, { id: compressToastId });
                });
                toast.success('预处理完成，体积大幅缩减！', { id: compressToastId });
            } catch (err) {
                console.warn('AI Pre-processing failed, gracefully falling back to original upload:', err);
                toast.error('本地加速暂不可用，正在为您切换为云端原生上传', { id: compressToastId, duration: 4000 });
                // 优雅降级：直接用原始视频上传，不要 return 拦截
                finalFile = f;
                setIsCompressing(false);
            }
            setIsCompressing(false);
        }

        // 2. 上传处理
        const tStart = Date.now();
        const toastId = toast.loading('Uploading video…');
        const tickHandle = setInterval(() => {
            const sec = Math.floor((Date.now() - tStart) / 1000);
            const mm = String(Math.floor(sec / 60)).padStart(2, '0');
            const ss = String(sec % 60).padStart(2, '0');
            toast.loading(`Uploading video… ${mm}:${ss} elapsed`, { id: toastId });
        }, 1000);
        try {
            const data = await uploadVideo(finalFile, (pct) => {
                setUploadPct(pct);
            });
            clearInterval(tickHandle);
            done();
            setUploadSuccess(true);
            setUploadedVideoId(data.session_id || data.video_id);
            toast.success('Upload complete', { id: toastId });
        } catch (err) {
            clearInterval(tickHandle);
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
        // Upload → MatchPeriods (Trimmer) → MultiSegmentConfig → Dashboard
        navigate(`/trim?sessionId=${encodeURIComponent(uploadedVideoId)}`, {
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
                                disabled={(!uploadSuccess && uploadPct > 0 && uploadPct < 100) || isCompressing}
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
                                        {!uploadSuccess && (
                                            isCompressing ? ` · AI pre-processing: ${compressPct}%` : ' · uploading…'
                                        )}
                                    </p>
                                </div>
                            </div>
                            <button
                                className={`btn ${uploadSuccess ? 'btn-success' : 'btn-primary'}`}
                                onClick={goToTrim}
                                disabled={!uploadSuccess || isCompressing}
                            >
                                {uploadSuccess ? 'Continue' : (isCompressing ? 'Pre-processing...' : 'Uploading…')}
                                <HiArrowRight />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
