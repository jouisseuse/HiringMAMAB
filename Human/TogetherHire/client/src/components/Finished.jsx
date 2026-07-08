import React from "react";

export function Finished() {
  return (
    <div className="h-full w-full flex items-center justify-center p-6">
      <div className="max-w-md text-center">
        <div className="text-2xl font-bold mb-2">All done 🎉</div>
        <p className="text-gray-600">Thanks for participating!</p>
      </div>
    </div>
  );
}