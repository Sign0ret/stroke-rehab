"use client";
import React, { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Button } from "@/components/ui/button";

export default function Motor() {
    const [mode, setMode] = useState<string>("pre-rehab");
    const [selectedTrainFile, setSelectedTrainFile] = useState<File | null>(null);
    const [selectedTestFile, setSelectedTestFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    // Function to handle file selection from FileUpload component
    const handleTrainFileChange = (files: File[]) => {
        if (files.length > 0) {
            setSelectedTrainFile(files[0]);  // Assuming only one file is being uploaded
        } else {
            setSelectedTrainFile(null);
        }
    };

    // Function to handle file selection from FileUpload component
    const handleTestFileChange = (files: File[]) => {
        if (files.length > 0) {
            setSelectedTestFile(files[0]);  // Assuming only one file is being uploaded
        } else {
            setSelectedTestFile(null);
        }
    };

    // Handle form submission and POST request to FastAPI endpoint
    const handleSubmit = async () => {
        setIsLoading(true);
        if (!selectedTrainFile) {
            alert("Please upload a train file before submitting.");
            return;
        }
        if (!selectedTestFile) {
            alert("Please upload a test file before submitting.");
            return;
        }
        console.log("mode:", mode);

        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append("trainfile", selectedTrainFile);
        formData.append("testfile", selectedTestFile);
        formData.append("mode", mode);  // Example of adding other form data

        console.log("selectedTrainFile:",selectedTrainFile)
        console.log("selectedTestFile:",selectedTestFile)

        try {
            const response = await fetch("http://127.0.0.1:8000/api/py/evaluate/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to submit data.");
            }

            const data = await response.json();
            console.log("Server response:", data);
            // You can handle the response here (e.g., display the results to the user)
        } catch (error) {
            console.error("Error:", error);
        }
        setIsLoading(false);
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-white text-black">
            <div className="max-w-2xl mx-auto p-4">
                <h1 className="relative z-10 text-lg md:text-7xl bg-clip-text text-transparent bg-gradient-to-b from-neutral-200 to-neutral-600 text-center font-sans font-bold">
                    Stroke Rehab Data Analysis
                </h1>
                <p></p>
                <p className="text-neutral-500 max-w-lg mx-auto my-2 text-sm text-center relative z-10">
                    Here you can upload your data to get accurate predictions...
                </p>
            </div>

            {/* Radio Group to select pre-rehab or post-rehab */}
            <RadioGroup value={mode} onValueChange={setMode}>
                <div className="flex items-center space-x-2">
                    <RadioGroupItem value="pre-rehab" id="pre-rehab" />
                    <Label htmlFor="pre-rehab">Pre Rehab</Label>
                </div>
                <div className="flex items-center space-x-2">
                    <RadioGroupItem value="post-rehab" id="post-rehab" />
                    <Label htmlFor="post-rehab">Post Rehab</Label>
                </div>
            </RadioGroup>
            <div className="flex flex-row">
                {/* File upload component */}
                <FileUpload onChange={handleTrainFileChange} title="train"/>

                {/* File upload component */}
                <FileUpload onChange={handleTestFileChange} title="test" />
            </div>

            {/* Button to submit */}
            <Button variant="default" className="" onClick={handleSubmit} disabled={isLoading}>
                Get Results
            </Button>
        </main>
    );
}