import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import TableContainer from "@mui/material/TableContainer";

const WeatherTable = ({ columns, rows }) => {
  return (
    <TableContainer sx={{ maxHeight: 400 }}>
      <Table stickyHeader aria-label="sticky table">
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
                        backgroundColor: "inherit",
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
            <TableRow hover role="checkbox" tabIndex={-1} key={rIndex}>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align="left"
                  sx={
                    column.id === "parameter"
                      ? {
                          position: "sticky",
                          left: 0,
                          backgroundColor: "inherit",
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
