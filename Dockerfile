# Use the official Node.js image as the base image
FROM node:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the package.json and package-lock.json to the container's working directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application files to the container's working directory
COPY . .

# Expose port 3000
EXPOSE 3000

# Set the NODE_ENV environment variable to 'development'
ENV NODE_ENV=development

# Start the Node.js application
CMD ["npm", "start"]