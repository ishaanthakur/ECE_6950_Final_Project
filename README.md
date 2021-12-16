# ECE 6950 Final Project
Name - Ishaan Thakur (it233) and Parth Sarthi Sharma (pss242)

Things to download/check before running the program:
<ol>

<li> Download yolo weights from  <a href="https://drive.google.com/drive/folders/18PZ6O-_zqMM5r4mwRf9KcYxyyMgmRvJ6?usp=sharing">here</a> and place it in the directory called <strong>yolo-coco</strong>.
<li> Ensure the system has Python version 3.7 with data processing libraries installed such as scipy, numpy and image processing library OpenCV.
<li> Connect rover and computer on which the program runs to the same WiFi network.
</ol>


How to run the program:
<ol>
<li> CD into project folder on the Martha rover and run the script <code>run.sh</code>.
<li> From the computer, execute the <code>Client.py</code> program and send the command <code>rs stream</code>.
<li> Clone this GitHub repository and cd into this directory.

<li>  Run the following command: <code>python3 StreamingClientFinal.py</code>.

</ol>


References:
Thanks to the following tutorial [here](https://www.youtube.com/watch?v=PTLZnE6W2tw&ab_channel=TheLazyCoder) for helping us how to setup yolo for person detection.
