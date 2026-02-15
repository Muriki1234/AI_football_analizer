import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './StepNav.css';

const STEPS = [
    { path: '/upload', label: 'Upload', num: 1 },
    { path: '/trim', label: 'Trim', num: 2 },
    { path: '/configure', label: 'Configure', num: 3 },
    { path: '/dashboard', label: 'Dashboard', num: 4 },
];

export default function StepNav() {
    const { pathname } = useLocation();
    const navigate = useNavigate();
    const currentIdx = STEPS.findIndex((s) => s.path === pathname);

    if (currentIdx < 0) return null;

    const handleClick = (step, i) => {
        // Allow clicking completed steps and the current step
        if (i <= currentIdx) {
            navigate(step.path);
        }
    };

    return (
        <nav className="step-nav">
            {STEPS.map((step, i) => {
                const done = i < currentIdx;
                const active = i === currentIdx;
                const clickable = i <= currentIdx;
                return (
                    <div key={step.path} className="step-nav__item">
                        <motion.div
                            className={`step-nav__circle ${done ? 'step-nav__circle--done' : ''} ${active ? 'step-nav__circle--active' : ''} ${clickable ? 'step-nav__circle--clickable' : ''}`}
                            initial={false}
                            animate={active ? { scale: [1, 1.15, 1] } : {}}
                            transition={{ duration: 0.4 }}
                            onClick={() => handleClick(step, i)}
                        >
                            {done ? '✓' : step.num}
                        </motion.div>
                        <span
                            className={`step-nav__label ${active ? 'step-nav__label--active' : ''} ${done ? 'step-nav__label--done' : ''} ${clickable ? 'step-nav__label--clickable' : ''}`}
                            onClick={() => handleClick(step, i)}
                        >
                            {step.label}
                        </span>
                        {i < STEPS.length - 1 && (
                            <div className={`step-nav__line ${done ? 'step-nav__line--done' : ''}`} />
                        )}
                    </div>
                );
            })}
        </nav>
    );
}
