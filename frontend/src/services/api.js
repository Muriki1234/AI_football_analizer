import axios from 'axios';

const api = axios.create({
    baseURL: '/api', // Proxy will handle the forwarding to backend
    headers: {
        'Content-Type': 'application/json',
    },
});

export default api;
