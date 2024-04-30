document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    const recordingStatus = document.getElementById('recordingStatus');
    let mediaRecorder;
    let audioChunks = [];

    recordButton.addEventListener('click', function() {
        if (recordButton.innerText === 'Record') {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/wav' }); // Specify WAV format
                    mediaRecorder.start();
                    recordButton.innerText = 'Stop Recording';
                    recordingStatus.innerText = 'Recording...';

                    mediaRecorder.ondataavailable = function(e) {
                        audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = function() {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' }); // Set type to audio/wav
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        // For playback, you may need to use alternative methods or libraries, 
                        // as not all browsers support WAV format directly with the Audio object.
                        // Example:
                        const audio = new Audio();
                        audio.src = audioUrl;
                        audio.controls = true;
                        document.body.appendChild(audio);
                        
                        audioChunks = [];
                        recordButton.innerText = 'Record';
                        recordingStatus.innerText = 'Not Recording';
                    };
                });
        } else {
            mediaRecorder.stop();
        }
    });
});
