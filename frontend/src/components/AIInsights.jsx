import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiSparkles, HiDocumentArrowDown, HiShare } from 'react-icons/hi2';
import './AIInsights.css';

const REPORT_TEXT = `## Tactical Analysis — Marcus R. (#10)

**Positioning:** Excellent positional awareness in the attacking third. Consistently finds space between the opposition's midfield and defensive lines.

**Sprint Burst Patterns:** 3 key sprints recorded in the 25th, 55th, and 75th minutes, each exceeding 30 km/h. Recovery time between sprints is within optimal range (< 45 seconds).

**Improvement Areas:**
1. **Defensive Transition:** When possession is lost, player takes an average of 3.2 seconds to begin pressing, compared to the team's ideal of 2.0 seconds. Recommend drills focused on reaction to lost possession.
2. **Left-side Coverage:** 78% of runs are through the right channel. Introducing training that encourages left-side movement will create better balance.
3. **Endurance Optimization:** Performance intensity drops by ~18% after the 70th minute. A targeted high-intensity interval training (HIIT) program is recommended.

**Overall Rating:** 8.2 / 10 — Above average performance with clear areas for tactical development.`;

export default function AIInsights() {
    const [generating, setGenerating] = useState(false);
    const [displayedText, setDisplayedText] = useState('');
    const [done, setDone] = useState(false);
    const indexRef = useRef(0);
    const containerRef = useRef(null);

    const generate = () => {
        setGenerating(true);
        setDisplayedText('');
        setDone(false);
        indexRef.current = 0;
    };

    useEffect(() => {
        if (!generating) return;
        const interval = setInterval(() => {
            indexRef.current += 3;
            if (indexRef.current >= REPORT_TEXT.length) {
                setDisplayedText(REPORT_TEXT);
                setDone(true);
                setGenerating(false);
                clearInterval(interval);
            } else {
                setDisplayedText(REPORT_TEXT.slice(0, indexRef.current));
            }
            if (containerRef.current) {
                containerRef.current.scrollTop = containerRef.current.scrollHeight;
            }
        }, 12);
        return () => clearInterval(interval);
    }, [generating]);

    return (
        <motion.div
            className="ai-panel card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
        >
            <div className="ai-panel__header">
                <HiSparkles className="ai-panel__sparkle" />
                <h3>AI Insights</h3>
            </div>

            <AnimatePresence mode="wait">
                {!displayedText && !generating ? (
                    <motion.div
                        key="empty"
                        className="ai-panel__empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <p>Click below to generate an AI-powered tactical report.</p>
                        <button className="btn btn-primary" onClick={generate}>
                            <HiSparkles /> Generate Report
                        </button>
                    </motion.div>
                ) : (
                    <motion.div
                        key="report"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div className="ai-panel__report" ref={containerRef}>
                            <pre className="ai-panel__text">{displayedText}</pre>
                            {generating && <span className="ai-panel__cursor" />}
                        </div>

                        {done && (
                            <motion.div
                                className="ai-panel__actions"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <button className="btn btn-secondary">
                                    <HiDocumentArrowDown /> Export PDF
                                </button>
                                <button className="btn btn-secondary">
                                    <HiShare /> Share
                                </button>
                            </motion.div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}
