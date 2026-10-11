import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useLanguage } from '../i18n/LanguageContext';
import './StepNav.css';

export default function StepNav() {
    const { pathname } = useLocation();
    const navigate = useNavigate();
    const { t } = useLanguage();

    const STEPS = [
        { path: '/upload', label: t('common.stepUpload'), num: 1 },
        { path: '/configure', label: t('common.stepConfigure'), num: 2 },
        { path: '/dashboard', label: t('common.stepDashboard'), num: 3 },
    ];
    // /configure-multi and /trim are both part of step 2 ("Configure"),
    // so the breadcrumb dot lights up consistently across the picker pages.
    const normalizedPath =
        pathname.startsWith('/configure') || pathname.startsWith('/trim')
            ? '/configure'
            : pathname;
    const currentIdx = STEPS.findIndex((s) => s.path === normalizedPath);

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
