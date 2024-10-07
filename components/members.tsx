"use client";
import React from "react";
import { AnimatedTooltip } from "./ui/animated-tooltip";
const people = [
  {
    id: 1,
    name: "Esteban Ochoa",
    designation: "ITC",
    image:
      "/simbolo-tec-black.png",
  },
  {
    id: 2,
    name: "Jose Emilio Inzunza",
    designation: "ITC",
    image:
        "/simbolo-tec-black.png",},
  {
    id: 3,
    name: "Roberto Morales",
    designation: "ITC",
    image:
        "/simbolo-tec-black.png",
},
  {
    id: 4,
    name: "Adolfo Hernández",
    designation: "ITC",
    image:
        "/simbolo-tec-black.png",
},
  {
    id: 5,
    name: "Esteban Muñoz",
    designation: "ITC",
    image:
      "/simbolo-tec-black.png",
  },
  {
    id: 6,
    name: "Alonso Rivera",
    designation: "IBM",
    image:
        "/simbolo-tec-black.png",
},
{
    id: 7,
    name: "Isabella Hurtado",
    designation: "IBM",
    image:
        "/simbolo-tec-black.png",
},
  
];

export function Members() {
  return (
    <div className="flex flex-row items-center justify-center mb-10 w-full">
      <AnimatedTooltip items={people} />
    </div>
  );
}
