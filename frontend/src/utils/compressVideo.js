import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';

let ffmpegInstance = null;
let isLoading = false;

const loadFFmpeg = async () => {
    if (ffmpegInstance) return ffmpegInstance;
    if (isLoading) {
        // Simple wait if another call is loading
        while (isLoading) await new Promise(r => setTimeout(r, 100));
        return ffmpegInstance;
    }
    
    isLoading = true;
    try {
        const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm';
        const ffmpeg = new FFmpeg();
        
        await ffmpeg.load({
            coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
            wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
            workerURL: await toBlobURL(`${baseURL}/ffmpeg-core.worker.js`, 'text/javascript'),
        });
        
        ffmpegInstance = ffmpeg;
        return ffmpegInstance;
    } finally {
        isLoading = false;
    }
};

/**
 * 压缩大于 threshold 的视频
 * @param {File} file - 原始视频文件
 * @param {Function} onProgress - 进度回调 (0-100)
 * @returns {Promise<File>} 压缩后的 File 对象，或原文件(如果不需压缩/压缩失败)
 */
export const compressVideoIfNeeded = async (file, onProgress) => {
    const SIZE_THRESHOLD = 500 * 1024 * 1024; // 500MB
    
    if (file.size <= SIZE_THRESHOLD) {
        console.log('File size under 500MB, skipping local compression.');
        return file;
    }
    
    console.log(`File size is ${(file.size / (1024*1024)).toFixed(1)}MB, starting local compression...`);
    
    let ffmpeg;
    try {
        ffmpeg = await loadFFmpeg();
        
        ffmpeg.on('progress', ({ progress, time }) => {
            // progress is usually 0 to 1
            const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
            if (onProgress) onProgress(pct);
        });

        const safeFileName = 'input_video.mp4'; // Avoid special characters in file.name causing issues
        
        // 使用 WORKERFS 挂载本地文件，避免内存溢出
        await ffmpeg.mount('WORKERFS', {
            files: [file],
        }, '/workerfs');
        
        // 确保输出文件名干净
        const outName = 'output.mp4';
        
        // 执行压缩：1080p，30fps，ultrafast 预设以最快速度压缩
        // -vf scale=-2:1080 保证宽度偶数，高度最高1080
        console.log('FFmpeg processing started...');
        await ffmpeg.exec([
            '-i', `/workerfs/${file.name}`,
            '-vf', 'scale=-2:1080,fps=30',
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'ultrafast',
            '-c:a', 'copy', // 直接复制音频，不耗费 CPU 重编音频
            outName
        ]);
        console.log('FFmpeg processing finished.');
        
        const data = await ffmpeg.readFile(outName);
        const blob = new Blob([data.buffer], { type: 'video/mp4' });
        
        // 提取原文件名去除后缀，加上 _compressed
        const baseName = file.name.replace(/\.[^/.]+$/, "");
        const compressedFile = new File([blob], `${baseName}_compressed.mp4`, { type: 'video/mp4' });
        
        // 清理工作区
        await ffmpeg.deleteFile(outName);
        await ffmpeg.unmount('/workerfs');
        
        console.log(`Compression success! New size: ${(compressedFile.size / (1024*1024)).toFixed(1)}MB`);
        return compressedFile;
    } catch (err) {
        console.error('Compression failed:', err);
        if (ffmpeg) {
            try { await ffmpeg.unmount('/workerfs'); } catch (e) {}
        }
        throw err;
    }
};
