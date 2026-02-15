import { useState, useEffect } from 'react';
import api from '../services/api';

function Home() {
    const [message, setMessage] = useState('');

    useEffect(() => {
        api.get('/test')
            .then(response => {
                setMessage(response.data.message);
            })
            .catch(error => {
                console.error('Error fetching data:', error);
                setMessage('Error connecting to backend');
            });
    }, []);

    return (
        <div style={{ textAlign: 'center', marginTop: '50px' }}>
            <h1>Welcome to the Fullstack App</h1>
            <p>Backend says: {message}</p>
        </div>
    );
}

export default Home;
