<!DOCTYPE html>
<html lang="en">

<body>

<h1>Material Stream Identification System (MSI)</h1>
<p><strong>Machine Learning Course – Fall 2025</strong><br>
Faculty of Computing and Artificial Intelligence, Cairo University</p>

<h2>Project Overview</h2>
<p>This repository contains a complete implementation of an automated Material Stream Identification system. The system classifies waste images into six material categories with an additional Unknown class using classical machine learning techniques.</p>

<h2>System Pipeline</h2>
<ol>
    <li>Data Augmentation</li>
    <li>Feature Extraction (ResNet50 – fixed descriptor)</li>
    <li>Classifier Training (SVM & k-NN)</li>
    <li>Unknown-Class Rejection</li>
    <li>Real-Time Camera Deployment</li>
</ol>

<h2>Material Classes</h2>
<table>
<tr><th>ID</th><th>Class</th></tr>
<tr><td>0</td><td>Glass</td></tr>
<tr><td>1</td><td>Paper</td></tr>
<tr><td>2</td><td>Cardboard</td></tr>
<tr><td>3</td><td>Plastic</td></tr>
<tr><td>4</td><td>Metal</td></tr>
<tr><td>5</td><td>Trash</td></tr>
<tr><td>6</td><td>Unknown</td></tr>
</table>

<h2>How to Run</h2>
<h3>1. Data Augmentation</h3>
<pre><code>python src/data_preparation/augment_data.py</code></pre>

<h3>2. Feature Extraction</h3>
<pre><code>python src/feature_extraction/feature_extraction.py</code></pre>

<h3>3. Train Models</h3>
<pre><code>python src/models/svm.py
python src/models/knn.py</code></pre>

<h3>4. Test with Images</h3>
<pre><code>python src/test_batch_imgs.py</code></pre>

<h3>5. Real-Time Deployment</h3>
<pre><code>python src/realtime_camera.py</code></pre>

<h2>Important Notes</h2>
<ul>
    <li>The CNN is used strictly as a fixed feature extractor.</li>
    <li>No deep learning fine-tuning is performed.</li>
    <li>Final classification relies on SVM and k-NN.</li>
</ul>

<h2>Dependencies</h2>
<ul>
    <li>Python 3.9+</li>
    <li>PyTorch</li>
    <li>Scikit-learn</li>
    <li>OpenCV</li>
    <li>Pandas, NumPy</li>
</ul>

<h2>License</h2>
<p>Academic use only.</p>

</body>
</html>

