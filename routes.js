const express = require('express');


// Create the router object
const router = express.Router();

router.get('/', (req, res) => {
    // Replace 'predictions' with the data you want to pass to the frontend
    const predictions = ['Prediction 1', 'Prediction 2', 'Prediction 3'];
    res.render('index', { predictions });
  });

module.exports = router