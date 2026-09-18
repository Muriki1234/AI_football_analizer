import '@testing-library/jest-dom';
import { vi } from 'vitest';

Object.defineProperty(HTMLCanvasElement.prototype, 'width', { get: () => 800, set: () => {} });
Object.defineProperty(HTMLCanvasElement.prototype, 'height', { get: () => 450, set: () => {} });

// Basic Canvas Mock
HTMLCanvasElement.prototype.getContext = () => ({
  clearRect: vi.fn(),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  fill: vi.fn(),
  stroke: vi.fn(),
  arc: vi.fn(),
  ellipse: vi.fn(),
  fillText: vi.fn(),
  measureText: () => ({ width: 0 }),
  scale: vi.fn(),
  drawImage: vi.fn(),
  setLineDash: vi.fn(),
  quadraticCurveTo: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  setTransform: vi.fn(),
  getBoundingClientRect: () => ({ top: 0, left: 0, bottom: 450, right: 800, width: 800, height: 450 }),
});
Element.prototype.getBoundingClientRect = () => ({ top: 0, left: 0, bottom: 450, right: 800, width: 800, height: 450 });
HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ top: 0, left: 0, bottom: 450, right: 800, width: 800, height: 450 });
HTMLCanvasElement.prototype.toDataURL = vi.fn(() => 'data:image/png;base64,');

global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

global.requestAnimationFrame = vi.fn((cb) => 1);
global.cancelAnimationFrame = vi.fn((id) => {});

vi.mock('react-icons/hi2', () => ({
  HiPencil: () => 'HiPencil',
  HiArrowUpRight: () => 'HiArrowUpRight',
  HiArrowUturnLeft: () => 'HiArrowUturnLeft',
  HiTrash: () => 'HiTrash',
  HiCamera: () => 'HiCamera',
  HiMinus: () => 'HiMinus',
  HiChartBar: () => 'HiChartBar',
  HiShieldCheck: () => 'HiShieldCheck',
  HiArrowsPointingIn: () => 'HiArrowsPointingIn',
  HiPlayCircle: () => 'HiPlayCircle',
  HiInformationCircle: () => 'HiInformationCircle',
}));
vi.mock('react-icons/fi', () => ({
  FiCircle: () => 'FiCircle',
}));

