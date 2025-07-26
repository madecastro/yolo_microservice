require('dotenv').config();
const mongoose = require('mongoose');

const MONGO_URI = process.env.MONGODB_URI;

if (!MONGO_URI) {
  console.error('❌ No MONGODB_URI found in environment variables');
  process.exit(1);
}

console.log('⏳ Attempting to connect to MongoDB...');

mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => {
  console.log('✅ Connected to MongoDB successfully');
  return mongoose.disconnect();
})
.then(() => {
  console.log('🔌 Disconnected cleanly');
  process.exit(0);
})
.catch(err => {
  console.error('❌ Connection failed:', err.message || err);
  process.exit(1);
});
