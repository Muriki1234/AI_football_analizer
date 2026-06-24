import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiSparkles, HiDocumentArrowDown, HiExclamationCircle } from 'react-icons/hi2';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { generateFeature, pollTaskStatus } from '../services/api';
import './AIInsights.css';

// Render Qwen's markdown output to HTML so ** ## - render as bold/heading/list
// instead of leaking through as literal characters in a <pre> block.
marked.setOptions({ gfm: true, breaks: true });

// XSS Defense: AI input comes from text in video frames (jersey/scoreboard etc.),
// an attacker could control video content to make LLM output `<img src=x onerror=...>`,
// which after marked.parse becomes executable HTML injected into the page (same-origin = stealing Supabase JWT).
const _renderMd = (md) => DOMPurify.sanitize(marked.parse(md || ''));

const STAGE_LABELS = {
    loading_data:        'Loading stats...',
    selecting_video:     'Selecting video source...',
    uploading_video:     'Uploading video to Gemini...',
    gemini_reasoning:    'Gemini is analyzing the match...',
    saving_report:       'Saving report...',
};

function stageLabel(stage) {
    if (!stage) return 'AI is analyzing...';
    if (STAGE_LABELS[stage]) return STAGE_LABELS[stage];
    if (stage.startsWith('gemini_processing_')) {
        const sec = stage.split('_').pop().replace('s', '');
        return `Gemini processing video ${sec}s...`;
    }
    return stage;
}

export default function AIInsights({ sessionId }) {
    const [phase, setPhase]           = useState('idle'); // idle | running | streaming | done | error
    const [progress, setProgress]     = useState(0);
    const [stage, setStage]           = useState('');
    const [errorMsg, setErrorMsg]     = useState(null);
    const [fullText, setFullText]     = useState('');
    const [displayedText, setShown]   = useState('');

    const indexRef     = useRef(0);
    const containerRef = useRef(null);

    // ── Trigger Generation ───────────────────────────────────────────
    const generate = async () => {
        if (!sessionId) {
            setErrorMsg('session_id missing');
            setPhase('error');
            return;
        }
        setPhase('running');
        setProgress(0);
        setStage('');
        setErrorMsg(null);
        setFullText('');
        setShown('');
        indexRef.current = 0;

        try {
            const { task_id } = await generateFeature(sessionId, 'ai_summary');
            const task = await pollTaskStatus(sessionId, task_id, (t) => {
                if (typeof t.progress === 'number') setProgress(t.progress);
                if (t.stage) setStage(t.stage);
            });
            const md = task?.result?.report_markdown || '';
            if (!md) throw new Error('Empty report');
            setFullText(md);
            setPhase('streaming');
        } catch (e) {
            console.error('[AIInsights] generate failed:', e);
            setErrorMsg(e.message || String(e));
            setPhase('error');
        }
    };

    // ── 打字机：full → displayed ──────────────────────────────────────
    useEffect(() => {
        if (phase !== 'streaming' || !fullText) return;
        const tick = setInterval(() => {
            indexRef.current += 4;
            if (indexRef.current >= fullText.length) {
                setShown(fullText);
                setPhase('done');
                clearInterval(tick);
            } else {
                setShown(fullText.slice(0, indexRef.current));
            }
            if (containerRef.current) {
                containerRef.current.scrollTop = containerRef.current.scrollHeight;
            }
        }, 15);
        return () => clearInterval(tick);
    }, [phase, fullText]);

    // ── 导出 MD ──────────────────────────────────────────────────────
    const exportMarkdown = () => {
        const blob = new Blob([fullText], { type: 'text/markdown;charset=utf-8' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `ai_summary_${sessionId || 'report'}.md`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };

    const isBusy = phase === 'running';

    return (
        <motion.div
            className="ai-panel card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
        >
            <div className="ai-panel__header">
                <HiSparkles className="ai-panel__sparkle" />
                <h3>AI Insights <span className="ai-panel__badge">Gemini 1.5</span></h3>
            </div>

            <AnimatePresence mode="wait">
                {phase === 'idle' && (
                    <motion.div
                        key="empty"
                        className="ai-panel__empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <p>Click the button below to generate a tactical analysis report using the tracked video and data.</p>
                        <button className="btn btn-primary" onClick={generate} disabled={!sessionId}>
                            <HiSparkles /> Generate Report
                        </button>
                    </motion.div>
                )}

                {phase === 'running' && (
                    <motion.div
                        key="loading"
                        className="ai-panel__empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <p>{stageLabel(stage)}</p>
                        <div style={{ marginTop: 12, width: '100%', background: '#2a2a3a',
                                       borderRadius: 6, height: 6, overflow: 'hidden' }}>
                            <div style={{
                                width: `${progress}%`, height: '100%',
                                background: 'linear-gradient(90deg, #00e59b, #3498db)',
                                transition: 'width 0.3s ease'
                            }}/>
                        </div>
                        <p style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>{progress}%</p>
                    </motion.div>
                )}

                {(phase === 'streaming' || phase === 'done') && (
                    <motion.div
                        key="report"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div className="ai-panel__report" ref={containerRef}>
                            <div
                                className="ai-panel__text ai-panel__markdown"
                                dangerouslySetInnerHTML={{ __html: _renderMd(displayedText) }}
                            />
                            {phase === 'streaming' && <span className="ai-panel__cursor" />}
                        </div>

                        {phase === 'done' && (
                            <motion.div
                                className="ai-panel__actions"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <button className="btn btn-secondary" onClick={exportMarkdown}>
                                    <HiDocumentArrowDown /> Export MD
                                </button>
                                <button className="btn btn-secondary" onClick={generate}>
                                    <HiSparkles /> Regenerate
                                </button>
                            </motion.div>
                        )}
                    </motion.div>
                )}

                {phase === 'error' && (
                    <motion.div
                        key="error"
                        className="ai-panel__empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <p style={{ color: '#e74c3c', display: 'flex', alignItems: 'center', gap: 6 }}>
                            <HiExclamationCircle /> {errorMsg}
                        </p>
                        <button className="btn btn-primary" onClick={generate}>
                            <HiSparkles /> Retry
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}
