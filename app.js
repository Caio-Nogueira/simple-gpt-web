const express = require('express');
const path = require("path")
const routes = require("./routes");

var app = express();

app.set("port", process.env.PORT || 3000);
app.use(routes)


app.listen(3000, () => {
    console.log("Server running on port 3000");
    }
);