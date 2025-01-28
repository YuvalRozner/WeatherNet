import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import TableContainer from "@mui/material/TableContainer";
import { useEffect, useState } from "react";

const WeatherTable = ({ dataset }) => {
  const [columns, setColumns] = useState([]);
  const [rows, setRows] = useState([]);

  // Build columns & rows for the transposed table
  useEffect(() => {
    if (dataset.length === 0) {
      setColumns([]);
      setRows([]);
      return;
    }

    const newColumns = [
      { id: "parameter", label: "Parameter", minWidth: 180 },
      ...dataset.map((item, index) => ({
        id: `time-${index}`,
        label: item.formattedTime,
        minWidth: 50,
      })),
    ];

    const paramRows = [
      { parameter: "Temperature (°C)", paramKey: "ImsTemp" },
      { parameter: "Rain Chance (%)", paramKey: "rain_chance" },
      { parameter: "Wave Height (m)", paramKey: "wave_height" },
      { parameter: "Relative Humidity (%)", paramKey: "relative_humidity" },
      { parameter: "Wind Speed (km/h)", paramKey: "wind_speed" },
    ];

    const newRows = paramRows.map((pRow) => {
      const rowObj = { parameter: pRow.parameter };
      dataset.forEach((item, idx) => {
        rowObj[`time-${idx}`] = item[pRow.paramKey] ?? "-";
      });
      return rowObj;
    });

    setColumns(newColumns);
    setRows(newRows);
  }, [dataset]);

  return (
    <TableContainer sx={{ maxHeight: 370, width: "90%" }}>
      <Table stickyHeader aria-label="sticky table" size="small">
        <TableHead>
          <TableRow>
            {columns.map((column) => (
              <TableCell
                key={column.id}
                align="left"
                style={{ minWidth: column.minWidth }}
                sx={
                  column.id === "parameter"
                    ? {
                        position: "sticky",
                        left: 0,
                        backgroundColor: "#282c34",
                        zIndex: 4,
                        boxShadow: "2px 0px 3px -1px rgba(0,0,0,0.1)",
                        fontWeight: "bold",
                      }
                    : {}
                }
              >
                {column.label}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row, rIndex) => (
            <TableRow
              hover
              role="checkbox"
              tabIndex={-1}
              key={rIndex}
              sx={{ height: 44 }}
            >
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align="left"
                  sx={
                    column.id === "parameter"
                      ? {
                          position: "sticky",
                          left: 0,
                          backgroundColor: "#282c34",
                          zIndex: 2,
                          boxShadow: "2px 0px 3px -1px rgba(0,0,0,0.1)",
                          fontWeight: "bold",
                        }
                      : {}
                  }
                >
                  {row[column.id]}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default WeatherTable;
