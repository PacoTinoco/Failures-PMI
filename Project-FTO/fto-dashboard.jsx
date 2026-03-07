'use client';

import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';

const FTO_DATA = {
  "R Vargas": {
    "coordinator": "R Vargas",
    "operators": [
      {
        "name": "Cristian Quintanilla",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#]", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 4.0, "BOS ENG [%]": 100.0, "QBOS [#]": 4.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 1.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 9.0, "DH Reparados [#]": 8.0, "Curva de Autonomía [%]": 89.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 0.0, "FRR [%]": 0.08, "DIM WASTE [%]": 3.34, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 100.0}},
          {"week": "2026-01-26", "values": {"BOS [#]": 10.0, "BOS ENG [%]": 100.0, "QBOS [#]": 6.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 2.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 28.0, "DH Reparados [#]": 27.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 1.0, "FRR [%]": 0.09, "DIM WASTE [%]": 3.34, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 100.0}},
          {"week": "2026-02-02", "values": {"BOS [#]": 4.0, "BOS ENG [%]": 100.0, "QBOS [#]": 5.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 20.0, "DH Reparados [#]": 16.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 0.0, "FRR [%]": 0.07, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 100.0}},
          {"week": "2026-02-09", "values": {"BOS [#]": 11.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 3.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 21.0, "DH Reparados [#]": 19.0, "Curva de Autonomía [%]": 90.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 0.0, "FRR [%]": 0.08, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 85.14}},
          {"week": "2026-02-16", "values": {"BOS [#]": 7.0, "BOS ENG [%]": 100.0, "QBOS [#]": 6.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 12.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 0.0, "FRR [%]": 0.11, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 85.14}},
          {"week": "2026-02-23", "values": {"BOS [#]": 6.0, "BOS ENG [%]": 100.0, "QBOS [#]": 4.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 1.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 0.0, "DH Reparados [#]": 8.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 1.0, "FRR [%]": 0.19, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 85.14}}
        ]
      },
      {
        "name": "Diana Magaña",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#] min 6", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 95.0, "QFlags [#] min 6": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 10.0, "DH Reparados [#]": 10.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 33.0, "BOS ENG [%]": 100.0, "QBOS [#]": 13.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 3.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 12.0, "DH Reparados [#]": 12.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.59, "DIM WASTE [%]": 4.82, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 66.67}},
          {"week": "2026-01-26", "values": {"BOS [#]": 26.0, "BOS ENG [%]": 100.0, "QBOS [#]": 11.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 23.0, "DH Reparados [#]": 23.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.07, "DIM WASTE [%]": 4.82, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 66.67}},
          {"week": "2026-02-02", "values": {"BOS [#]": 31.0, "BOS ENG [%]": 100.0, "QBOS [#]": 11.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 2.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 29.0, "DH Reparados [#]": 28.0, "Curva de Autonomía [%]": 97.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 0.0, "FRR [%]": 0.0, "DIM WASTE [%]": 0.86, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 2.0, "QM On Target [%]": 66.67}},
          {"week": "2026-02-09", "values": {"BOS [#]": 39.0, "BOS ENG [%]": 100.0, "QBOS [#]": 19.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 12.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 33.0, "DH Reparados [#]": 34.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 85.0, "IPS [#]": 1.0, "FRR [%]": 0.47, "DIM WASTE [%]": 0.86, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 66.67}},
          {"week": "2026-02-16", "values": {"BOS [#]": 34.0, "BOS ENG [%]": 100.0, "QBOS [#]": 19.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 29.0, "DH Reparados [#]": 28.0, "Curva de Autonomía [%]": 97.0, "Contramedidas Defectos [%]": 10.0, "IPS [#]": 1.0, "FRR [%]": 0.27, "DIM WASTE [%]": 0.86, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 66.67}},
          {"week": "2026-02-23", "values": {"BOS [#]": 39.0, "BOS ENG [%]": 100.0, "QBOS [#]": 15.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 10.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 21.0, "DH Reparados [#]": 22.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 86.0, "IPS [#]": 1.0, "FRR [%]": 0.37, "DIM WASTE [%]": 0.86, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 66.67}}
        ]
      },
      {
        "name": "Luis Ramírez",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#] min 6", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#] min 6": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-02-09", "values": {"BOS [#]": 3.0, "BOS ENG [%]": 100.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 7.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 11.0, "DH Reparados [#]": 8.0, "Curva de Autonomía [%]": 73.0, "Contramedidas Defectos [%]": 50.0, "IPS [#]": 0.0, "FRR [%]": 1.45, "DIM WASTE [%]": 1.12, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 50.0}},
          {"week": "2026-02-16", "values": {"BOS [#]": 3.0, "BOS ENG [%]": 100.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 1.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 0.0, "DH Reparados [#]": 0.0, "Curva de Autonomía [%]": 0.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 0.0, "FRR [%]": 0.75, "DIM WASTE [%]": 1.12, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 50.0}},
          {"week": "2026-02-23", "values": {"BOS [#]": 3.0, "BOS ENG [%]": 100.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 100.0, "QFlags [#] min 6": 2.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 1.0, "DH Reparados [#]": 0.0, "Curva de Autonomía [%]": 0.0, "Contramedidas Defectos [%]": 0.0, "IPS [#]": 0.0, "FRR [%]": 0.82, "DIM WASTE [%]": 1.12, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 50.0}}
        ]
      }
    ]
  },
  "C Lara": {
    "coordinator": "C Lara",
    "operators": [
      {
        "name": "Jorge Molina",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#]", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 5.0, "BOS ENG [%]": 100.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 2.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 18.0, "DH Reparados [#]": 16.0, "Curva de Autonomía [%]": 89.0, "Contramedidas Defectos [%]": 40.0, "IPS [#]": 1.0, "FRR [%]": 0.22, "DIM WASTE [%]": 2.95, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 75.0}},
          {"week": "2026-01-26", "values": {"BOS [#]": 12.0, "BOS ENG [%]": 100.0, "QBOS [#]": 8.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 5.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 24.0, "DH Reparados [#]": 23.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 60.0, "IPS [#]": 1.0, "FRR [%]": 0.10, "DIM WASTE [%]": 2.95, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 75.0}},
          {"week": "2026-02-02", "values": {"BOS [#]": 8.0, "BOS ENG [%]": 100.0, "QBOS [#]": 6.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 3.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 22.0, "DH Reparados [#]": 20.0, "Curva de Autonomía [%]": 91.0, "Contramedidas Defectos [%]": 70.0, "IPS [#]": 0.0, "FRR [%]": 0.05, "DIM WASTE [%]": 0.57, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 75.0}},
          {"week": "2026-02-09", "values": {"BOS [#]": 15.0, "BOS ENG [%]": 100.0, "QBOS [#]": 7.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 8.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 28.0, "DH Reparados [#]": 27.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 75.0, "IPS [#]": 1.0, "FRR [%]": 0.16, "DIM WASTE [%]": 0.57, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 75.0}},
          {"week": "2026-02-16", "values": {"BOS [#]": 10.0, "BOS ENG [%]": 100.0, "QBOS [#]": 8.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 16.0, "DH Reparados [#]": 15.0, "Curva de Autonomía [%]": 94.0, "Contramedidas Defectos [%]": 80.0, "IPS [#]": 0.0, "FRR [%]": 0.08, "DIM WASTE [%]": 0.57, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 75.0}},
          {"week": "2026-02-23", "values": {"BOS [#]": 9.0, "BOS ENG [%]": 100.0, "QBOS [#]": 7.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 19.0, "DH Reparados [#]": 19.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 85.0, "IPS [#]": 1.0, "FRR [%]": 0.14, "DIM WASTE [%]": 0.57, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 75.0}}
        ]
      },
      {
        "name": "Fabián López",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#]", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 4.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 1.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 8.0, "DH Reparados [#]": 7.0, "Curva de Autonomía [%]": 88.0, "Contramedidas Defectos [%]": 50.0, "IPS [#]": 0.0, "FRR [%]": 0.12, "DIM WASTE [%]": 3.07, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 90.0}},
          {"week": "2026-01-26", "values": {"BOS [#]": 11.0, "BOS ENG [%]": 100.0, "QBOS [#]": 5.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 3.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 20.0, "DH Reparados [#]": 19.0, "Curva de Autonomía [%]": 95.0, "Contramedidas Defectos [%]": 70.0, "IPS [#]": 1.0, "FRR [%]": 0.09, "DIM WASTE [%]": 3.07, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 90.0}},
          {"week": "2026-02-02", "values": {"BOS [#]": 7.0, "BOS ENG [%]": 100.0, "QBOS [#]": 4.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 2.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 19.0, "DH Reparados [#]": 17.0, "Curva de Autonomía [%]": 89.0, "Contramedidas Defectos [%]": 80.0, "IPS [#]": 0.0, "FRR [%]": 0.06, "DIM WASTE [%]": 0.71, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 90.0}},
          {"week": "2026-02-09", "values": {"BOS [#]": 14.0, "BOS ENG [%]": 100.0, "QBOS [#]": 6.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 5.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 25.0, "DH Reparados [#]": 24.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 85.0, "IPS [#]": 1.0, "FRR [%]": 0.13, "DIM WASTE [%]": 0.71, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 90.0}},
          {"week": "2026-02-16", "values": {"BOS [#]": 9.0, "BOS ENG [%]": 100.0, "QBOS [#]": 7.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 17.0, "DH Reparados [#]": 16.0, "Curva de Autonomía [%]": 94.0, "Contramedidas Defectos [%]": 90.0, "IPS [#]": 0.0, "FRR [%]": 0.08, "DIM WASTE [%]": 0.71, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 90.0}},
          {"week": "2026-02-23", "values": {"BOS [#]": 8.0, "BOS ENG [%]": 100.0, "QBOS [#]": 6.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 5.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 18.0, "DH Reparados [#]": 18.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 95.0, "IPS [#]": 1.0, "FRR [%]": 0.11, "DIM WASTE [%]": 0.71, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 90.0}}
        ]
      }
    ]
  },
  "M Vidrio": {
    "coordinator": "M Vidrio",
    "operators": [
      {
        "name": "José García",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#]", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 85.0, "Contramedidas Defectos [%]": 95.0, "IPS [#]": 0.0, "FRR [%]": 0.08, "DIM WASTE [%]": 3.45, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0}},
          {"week": "2026-01-26", "values": {"BOS [#]": 8.0, "BOS ENG [%]": 100.0, "QBOS [#]": 4.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 7.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 22.0, "DH Reparados [#]": 21.0, "Curva de Autonomía [%]": 98.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.10, "DIM WASTE [%]": 3.45, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 90.0}},
          {"week": "2026-02-02", "values": {"BOS [#]": 5.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 5.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 18.0, "DH Reparados [#]": 17.0, "Curva de Autonomía [%]": 89.0, "Contramedidas Defectos [%]": 98.0, "IPS [#]": 0.0, "FRR [%]": 0.06, "DIM WASTE [%]": 0.59, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 85.0}},
          {"week": "2026-02-09", "values": {"BOS [#]": 12.0, "BOS ENG [%]": 100.0, "QBOS [#]": 5.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 9.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 26.0, "DH Reparados [#]": 25.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 99.0, "IPS [#]": 1.0, "FRR [%]": 0.12, "DIM WASTE [%]": 0.59, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 88.0}},
          {"week": "2026-02-16", "values": {"BOS [#]": 8.0, "BOS ENG [%]": 100.0, "QBOS [#]": 4.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 7.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 19.0, "DH Reparados [#]": 18.0, "Curva de Autonomía [%]": 92.0, "Contramedidas Defectos [%]": 97.0, "IPS [#]": 0.0, "FRR [%]": 0.09, "DIM WASTE [%]": 0.59, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 85.0}},
          {"week": "2026-02-23", "values": {"BOS [#]": 7.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 8.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 20.0, "DH Reparados [#]": 20.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.10, "DIM WASTE [%]": 0.59, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 88.0}}
        ]
      },
      {
        "name": "Andrés Fernández",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#]", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 4.0, "BOS ENG [%]": 100.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 2.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 12.0, "DH Reparados [#]": 11.0, "Curva de Autonomía [%]": 92.0, "Contramedidas Defectos [%]": 92.0, "IPS [#]": 1.0, "FRR [%]": 0.10, "DIM WASTE [%]": 3.12, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 85.0}},
          {"week": "2026-01-26", "values": {"BOS [#]": 9.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 4.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 19.0, "DH Reparados [#]": 18.0, "Curva de Autonomía [%]": 97.0, "Contramedidas Defectos [%]": 96.0, "IPS [#]": 0.0, "FRR [%]": 0.08, "DIM WASTE [%]": 3.12, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 87.0}},
          {"week": "2026-02-02", "values": {"BOS [#]": 6.0, "BOS ENG [%]": 100.0, "QBOS [#]": 2.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 3.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 16.0, "DH Reparados [#]": 15.0, "Curva de Autonomía [%]": 94.0, "Contramedidas Defectos [%]": 99.0, "IPS [#]": 0.0, "FRR [%]": 0.05, "DIM WASTE [%]": 0.65, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 88.0}},
          {"week": "2026-02-09", "values": {"BOS [#]": 13.0, "BOS ENG [%]": 100.0, "QBOS [#]": 4.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 24.0, "DH Reparados [#]": 23.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 98.0, "IPS [#]": 1.0, "FRR [%]": 0.11, "DIM WASTE [%]": 0.65, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 89.0}},
          {"week": "2026-02-16", "values": {"BOS [#]": 9.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 5.0, "QI / PNC [%]": 0.0, "DH encontrados [#]": 17.0, "DH Reparados [#]": 16.0, "Curva de Autonomía [%]": 94.0, "Contramedidas Defectos [%]": 96.0, "IPS [#]": 0.0, "FRR [%]": 0.07, "DIM WASTE [%]": 0.65, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 87.0}},
          {"week": "2026-02-23", "values": {"BOS [#]": 8.0, "BOS ENG [%]": 100.0, "QBOS [#]": 3.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 7.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 18.0, "DH Reparados [#]": 18.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 97.0, "IPS [#]": 1.0, "FRR [%]": 0.09, "DIM WASTE [%]": 0.65, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 0.0, "QM On Target [%]": 88.0}}
        ]
      }
    ]
  },
  "LC": {
    "coordinator": "Line Coordinator",
    "operators": [
      {
        "name": "Consolidado",
        "kpis": ["BOS [#]", "BOS ENG [%]", "QBOS [#]", "QBOS ENG [%]", "QFlags [#]", "QI / PNC [#]", "DH encontrados [#]", "DH Reparados [#]", "Curva de Autonomía [%]", "Contramedidas Defectos [%]", "IPS [#]", "FRR [%]", "DIM WASTE [%]", "Sobrepeso [#]", "Eventos LAIKA [#]", "Casos de estudios [#]", "QM On Target [%]"],
        "categories": ["Sustentabilidad", "Sustentabilidad", "Calidad", "Calidad", "Calidad", "Calidad", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Desempeño", "Costo", "Costo", "Costo", "Costo", "Moral", "Moral"],
        "objectives": {"BOS [#]": 3.0, "BOS ENG [%]": 95.0, "QBOS [#]": 1.0, "QBOS ENG [%]": 95.0, "QFlags [#]": 6.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 14.0, "DH Reparados [#]": 14.0, "Curva de Autonomía [%]": 80.0, "Contramedidas Defectos [%]": 100.0, "IPS [#]": 1.0, "FRR [%]": 0.1, "DIM WASTE [%]": 0.0, "Sobrepeso [#]": -20.0, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 80.0},
        "weekly_data": [
          {"week": "2026-01-19", "values": {"BOS [#]": 53.0, "BOS ENG [%]": 100.0, "QBOS [#]": 22.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 15.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 58.0, "DH Reparados [#]": 54.0, "Curva de Autonomía [%]": 91.0, "Contramedidas Defectos [%]": 80.0, "IPS [#]": 4.0, "FRR [%]": 0.25, "DIM WASTE [%]": 3.34, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 86.0}},
          {"week": "2026-01-26", "values": {"BOS [#]": 57.0, "BOS ENG [%]": 100.0, "QBOS [#]": 25.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 12.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 95.0, "DH Reparados [#]": 90.0, "Curva de Autonomía [%]": 96.0, "Contramedidas Defectos [%]": 85.0, "IPS [#]": 4.0, "FRR [%]": 0.22, "DIM WASTE [%]": 3.34, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 86.0}},
          {"week": "2026-02-02", "values": {"BOS [#]": 46.0, "BOS ENG [%]": 100.0, "QBOS [#]": 23.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 12.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 89.0, "DH Reparados [#]": 81.0, "Curva de Autonomía [%]": 91.0, "Contramedidas Defectos [%]": 90.0, "IPS [#]": 0.0, "FRR [%]": 0.18, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 3.0, "QM On Target [%]": 88.0}},
          {"week": "2026-02-09", "values": {"BOS [#]": 79.0, "BOS ENG [%]": 100.0, "QBOS [#]": 31.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 25.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 113.0, "DH Reparados [#]": 108.0, "Curva de Autonomía [%]": 94.0, "Contramedidas Defectos [%]": 88.0, "IPS [#]": 3.0, "FRR [%]": 0.26, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 84.0}},
          {"week": "2026-02-16", "values": {"BOS [#]": 62.0, "BOS ENG [%]": 100.0, "QBOS [#]": 28.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 19.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 74.0, "DH Reparados [#]": 73.0, "Curva de Autonomía [%]": 94.0, "Contramedidas Defectos [%]": 87.0, "IPS [#]": 1.0, "FRR [%]": 0.20, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 82.0}},
          {"week": "2026-02-23", "values": {"BOS [#]": 63.0, "BOS ENG [%]": 100.0, "QBOS [#]": 26.0, "QBOS ENG [%]": 100.0, "QFlags [#]": 22.0, "QI / PNC [#]": 0.0, "DH encontrados [#]": 58.0, "DH Reparados [#]": 66.0, "Curva de Autonomía [%]": 100.0, "Contramedidas Defectos [%]": 91.0, "IPS [#]": 4.0, "FRR [%]": 0.24, "DIM WASTE [%]": 0.47, "Eventos LAIKA [#]": 0.0, "Casos de estudios [#]": 1.0, "QM On Target [%]": 81.0}}
        ]
      }
    ]
  }
};

const CATEGORY_NAMES = {
  "Sustentabilidad": "S",
  "Calidad": "Q",
  "Desempeño": "D",
  "Costo": "C",
  "Moral": "M"
};

const CATEGORY_COLORS = {
  "Sustentabilidad": "#10B981",
  "Calidad": "#3B82F6",
  "Desempeño": "#F59E0B",
  "Costo": "#EF4444",
  "Moral": "#8B5CF6"
};

const getStatus = (value, objective) => {
  if (objective === 0) return 'neutral';
  if (objective > 0) {
    if (value >= objective * 0.95) return 'green';
    if (value >= objective * 0.85) return 'yellow';
    return 'red';
  } else {
    if (value <= objective * 1.05) return 'green';
    if (value <= objective * 1.15) return 'yellow';
    return 'red';
  }
};

const StatusBadge = ({ status }) => {
  const colors = {
    green: 'bg-emerald-900 border-emerald-700',
    yellow: 'bg-amber-900 border-amber-700',
    red: 'bg-red-900 border-red-700',
    neutral: 'bg-slate-700 border-slate-600'
  };
  return <div className={`w-4 h-4 rounded-full border ${colors[status]}`} />;
};

const KPICard = ({ name, value, objective, category }) => {
  const status = getStatus(value, objective);
  const displayValue = typeof value === 'number' ? value.toFixed(2) : value;
  const displayObjective = typeof objective === 'number' ? objective.toFixed(2) : objective;

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-md hover:border-slate-600 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-400 truncate">{name}</p>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-2xl font-bold text-slate-100">{displayValue}</span>
            <span className="text-xs text-slate-500">/ {displayObjective}</span>
          </div>
        </div>
        <StatusBadge status={status} />
      </div>
      <div className="flex items-center gap-2 mt-3">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: CATEGORY_COLORS[category] }}
        />
        <span className="text-xs font-semibold text-slate-400">{CATEGORY_NAMES[category]}</span>
      </div>
    </div>
  );
};

const TrendChart = ({ title, data, kpis, categoryColor, objectives }) => {
  const chartData = data.map(week => ({
    week: week.week,
    ...kpis.reduce((acc, kpi) => {
      acc[kpi] = week.values[kpi] || 0;
      return acc;
    }, {})
  }));

  const colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6'];

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-md">
      <h3 className="text-sm font-semibold text-slate-200 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="week" stroke="#78909C" style={{ fontSize: '12px' }} />
          <YAxis stroke="#78909C" style={{ fontSize: '12px' }} />
          <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #475569', borderRadius: '8px' }} />
          <Legend wrapperStyle={{ paddingTop: '10px' }} />
          {kpis.map((kpi, idx) => (
            <Line key={kpi} type="monotone" dataKey={kpi} stroke={colors[idx % colors.length]} strokeWidth={2} dot={false} />
          ))}
          {kpis.map((kpi) => (
            <ReferenceLine key={`ref-${kpi}`} y={objectives[kpi]} stroke={CATEGORY_COLORS["Calidad"]} strokeDasharray="5 5" opacity={0.3} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const ComparisonChart = ({ title, data, kpi, objective }) => {
  const colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6'];

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-md">
      <h3 className="text-sm font-semibold text-slate-200 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" stroke="#78909C" style={{ fontSize: '12px' }} />
          <YAxis stroke="#78909C" style={{ fontSize: '12px' }} />
          <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #475569', borderRadius: '8px' }} />
          <ReferenceLine y={objective} stroke="#EF4444" strokeDasharray="5 5" label={{ value: 'Objetivo', position: 'insideTopRight', offset: -10 }} />
          <Bar dataKey="value" fill={colors[0]} radius={[4, 4, 0, 0]}>
            {data.map((entry, idx) => (
              <Cell key={`cell-${idx}`} fill={colors[idx % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default function FTODashboard() {
  const [selectedCoordinator, setSelectedCoordinator] = useState('R Vargas');
  const [selectedOperator, setSelectedOperator] = useState(null);
  const [selectedKPI, setSelectedKPI] = useState(null);

  const coordinators = Object.keys(FTO_DATA);
  const operatorData = FTO_DATA[selectedCoordinator];

  const currentOperator = useMemo(() => {
    if (!selectedOperator) {
      setSelectedOperator(operatorData.operators[0].name);
      return operatorData.operators[0];
    }
    return operatorData.operators.find(op => op.name === selectedOperator) || operatorData.operators[0];
  }, [selectedCoordinator, selectedOperator, operatorData]);

  const kpisForCategory = (category) => {
    return currentOperator.kpis.filter((_, idx) => currentOperator.categories[idx] === category);
  };

  const latestValues = useMemo(() => {
    const latest = currentOperator.weekly_data[currentOperator.weekly_data.length - 1]?.values || {};
    return latest;
  }, [currentOperator]);

  const comparisonData = useMemo(() => {
    if (!selectedKPI) return [];
    const latestWeek = operatorData.operators[0].weekly_data[operatorData.operators[0].weekly_data.length - 1];
    return operatorData.operators.map(op => {
      const opLatestWeek = op.weekly_data[op.weekly_data.length - 1];
      return {
        name: op.name,
        value: opLatestWeek?.values[selectedKPI] || 0
      };
    });
  }, [selectedKPI, operatorData]);

  const handleSelectKPI = (kpi) => {
    setSelectedKPI(selectedKPI === kpi ? null : kpi);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 overflow-x-hidden">
      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; }
      `}</style>

      <div className="bg-slate-800 border-b border-slate-700 shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">FTO Dashboard - Filtros Blancos 1</h1>
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-400">Máquina:</label>
              <select
                value={selectedCoordinator}
                onChange={(e) => setSelectedCoordinator(e.target.value)}
                className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-slate-500"
              >
                {coordinators.map(coord => (
                  <option key={coord} value={coord}>{coord}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1 overflow-x-auto">
            {coordinators.map(coord => (
              <button
                key={coord}
                onClick={() => setSelectedCoordinator(coord)}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                  selectedCoordinator === coord
                    ? 'border-emerald-500 text-emerald-400'
                    : 'border-transparent text-slate-400 hover:text-slate-300'
                }`}
              >
                {coord}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="mb-6">
          <h2 className="text-sm font-semibold text-slate-400 mb-3">Operador</h2>
          <div className="flex flex-wrap gap-2">
            {operatorData.operators.map(op => (
              <button
                key={op.name}
                onClick={() => setSelectedOperator(op.name)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedOperator === op.name
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {op.name}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-8">
          {['Sustentabilidad', 'Calidad', 'Desempeño', 'Costo', 'Moral'].map(category => {
            const categoryKPIs = kpisForCategory(category);
            return (
              <div key={category}>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: CATEGORY_COLORS[category] }} />
                  <h3 className="text-lg font-semibold text-slate-200">{category} ({CATEGORY_NAMES[category]})</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {categoryKPIs.map(kpi => (
                    <button
                      key={kpi}
                      onClick={() => handleSelectKPI(kpi)}
                      className="text-left"
                    >
                      <KPICard
                        name={kpi}
                        value={latestValues[kpi] || 0}
                        objective={currentOperator.objectives[kpi]}
                        category={category}
                      />
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {['Sustentabilidad', 'Calidad', 'Desempeño', 'Costo', 'Moral'].map(category => {
            const categoryKPIs = kpisForCategory(category);
            if (categoryKPIs.length === 0) return null;
            return (
              <TrendChart
                key={category}
                title={`Tendencia - ${category}`}
                data={currentOperator.weekly_data}
                kpis={categoryKPIs}
                categoryColor={CATEGORY_COLORS[category]}
                objectives={currentOperator.objectives}
              />
            );
          })}
        </div>

        {selectedKPI && (
          <div className="mt-8">
            <ComparisonChart
              title={`Comparación: ${selectedKPI}`}
              data={comparisonData}
              kpi={selectedKPI}
              objective={currentOperator.objectives[selectedKPI] || 0}
            />
          </div>
        )}

        <div className="mt-8 bg-slate-800 border border-slate-700 rounded-lg overflow-hidden shadow-md">
          <div className="px-6 py-4 border-b border-slate-700">
            <h3 className="text-lg font-semibold text-slate-200">Datos Semanales - {currentOperator.name}</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-700/50">
                  <th className="px-4 py-3 text-left font-semibold text-slate-300">Semana</th>
                  {currentOperator.kpis.map(kpi => (
                    <th key={kpi} className="px-4 py-3 text-right font-semibold text-slate-300">{kpi}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {currentOperator.weekly_data.map((week, idx) => (
                  <tr key={idx} className="border-b border-slate-700 hover:bg-slate-700/30 transition-colors">
                    <td className="px-4 py-3 font-medium text-slate-300">{week.week}</td>
                    {currentOperator.kpis.map(kpi => (
                      <td key={kpi} className="px-4 py-3 text-right text-slate-400">
                        {(week.values[kpi] ?? '-').toFixed ? (week.values[kpi] || 0).toFixed(2) : week.values[kpi] ?? '-'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
