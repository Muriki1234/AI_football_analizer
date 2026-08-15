import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const url = process.env.SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_KEY;
const supabase = createClient(url, key);

async function main() {
  const { data, error } = await supabase
    .from('sessions')
    .select('id, video_url, created_at, status, progress, extra')
    .order('created_at', { ascending: false })
    .limit(1);
    
  if (error) {
    console.error(error);
  } else {
    console.log(JSON.stringify(data[0], null, 2));
  }
}

main();
