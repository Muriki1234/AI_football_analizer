const MIN_PERIOD_FOR_MULTI_SEG = 5;
const TOTAL_SEGS = 11;

function distributeSegments(periodsFr, fps) {
    const totalSec = periodsFr.reduce((sum, pr) => sum + (pr.endFrame - pr.startFrame) / fps, 0);
    const counts = periodsFr.map(pr => {
        const periodSec = (pr.endFrame - pr.startFrame) / fps;
        if (!Number.isFinite(periodSec) || periodSec < MIN_PERIOD_FOR_MULTI_SEG) return 1;
        return Math.max(1, Math.round((periodSec / totalSec) * TOTAL_SEGS));
    });

    while (counts.reduce((a, b) => a + b, 0) !== TOTAL_SEGS) {
        const sum = counts.reduce((a, b) => a + b, 0);
        if (sum < TOTAL_SEGS) {
            const maxIdx = periodsFr.reduce((maxI, pr, i, arr) => 
                (pr.endFrame - pr.startFrame) > (arr[maxI].endFrame - arr[maxI].startFrame) ? i : maxI, 0);
            counts[maxIdx]++;
        } else {
            const maxIdx = counts.reduce((maxI, count, i, arr) => (count > 1 && count > arr[maxI]) ? i : maxI, 0);
            counts[maxIdx]--;
        }
    }
    return counts;
}

console.log(distributeSegments([{startFrame: 0, endFrame: 5400 * 25}], 25));
